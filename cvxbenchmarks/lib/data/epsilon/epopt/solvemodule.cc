#include <Python.h>

#include <setjmp.h>
#include <stdlib.h>

#include <glog/logging.h>

#include "epsilon/algorithms/prox_admm.h"
#include "epsilon/algorithms/prox_admm_two_block.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/util/logging.h"
#include "epsilon/vector/vector_util.h"

// TODO(mwytock): Does failure handling need to be made threadsafe? Seems like
// making these threadlocal would do
static jmp_buf failure_buf;
static PyObject* SolveError;

std::unordered_map<std::string, std::unique_ptr<Solver>> global_solver_cache;

BlockVector GetVariableVector(PyObject* vars) {
  BlockVector x;

  Py_ssize_t pos = 0;
  PyObject* key;
  PyObject* value;
  while (PyDict_Next(vars, &pos, &key, &value)) {
    const char* key_str = PyString_AsString(key);
    char* value_str;
    Py_ssize_t value_str_len;
    PyString_AsStringAndSize(value, &value_str, &value_str_len);
    CHECK(key_str != nullptr);
    CHECK(value_str != nullptr);
    CHECK(value_str_len % sizeof(double) == 0);

    x(key_str) = Eigen::Map<const Eigen::VectorXd>(
        reinterpret_cast<const double*>(value_str),
        value_str_len/sizeof(double));
  }
  return x;
}

PyObject* GetVariableMap(const BlockVector& x) {
  PyObject* vars = PyDict_New();
  for (auto iter : x.data()) {
    PyObject* val = PyString_FromStringAndSize(
        reinterpret_cast<const char*>(iter.second.data()),
        iter.second.rows()*sizeof(double));
    PyDict_SetItemString(vars, iter.first.c_str(), val);
    Py_DECREF(val);
  }
  return vars;
}

void WriteConstants(PyObject* data, DataMap* data_map) {
  // NOTE(mwytock): References returned by PyDict_Next() are borrowed so no need
  // to Py_DECREF() them.
  Py_ssize_t pos = 0;
  PyObject* key;
  PyObject* value;
  while (PyDict_Next(data, &pos, &key, &value)) {
    const char* key_str = PyString_AsString(key);
    const char* value_str = PyString_AsString(value);
    CHECK(key_str);
    CHECK(value_str);
    data_map->insert(
        std::make_pair(key_str, std::string(value_str, PyString_Size(value))));
  }
}

std::unique_ptr<Solver> CreateSolver(
    const Problem& problem,
    const DataMap& data_map,
    const SolverParams& params) {
  if (params.solver() == SolverParams::PROX_ADMM) {
    return std::unique_ptr<Solver>(
        new ProxADMMSolver(problem, data_map, params));
  } else if (params.solver() == SolverParams::PROX_ADMM_TWO_BLOCK) {
    return std::unique_ptr<Solver>(
        new ProxADMMTwoBlockSolver(problem, data_map, params));
  } else {
    LOG(FATAL) << "Unknown solver: " << params.solver();
  }
}

void SetParameterValues(PyObject* parameters, Solver* solver) {
  PyObject* iter = PyObject_GetIter(parameters);
  PyObject* item ;
  CHECK(iter != nullptr);
  while ((item = PyIter_Next(iter))) {
    const char* parameter_id = PyString_AsString(PyTuple_GetItem(item, 0));

    Constant constant;
    PyObject* val = PyTuple_GetItem(item, 1);
    CHECK(constant.ParseFromArray(PyString_AsString(val), PyString_Size(val)));

    solver->SetParameterValue(parameter_id, constant);
    Py_DECREF(item);
  }

  Py_DECREF(iter);
  CHECK(!PyErr_Occurred());
}

extern "C" {

static PyObject* Solve(PyObject* self, PyObject* args) {
  const char* problem_str;
  const char* solver_params_str;
  int problem_str_len, solver_params_str_len;
  PyObject* data;
  PyObject* parameters;

  // prox_admm_solve(problem_str, params_str, data)
  if (!PyArg_ParseTuple(
          args, "s#Os#O",
          &problem_str, &problem_str_len,
          &parameters,
          &solver_params_str, &solver_params_str_len,
          &data)) {
    // TODO(mwytock): Need to set the appropriate exceptions when passed
    // incorrect arguments.
    return nullptr;
  }

  Problem problem;
  SolverParams solver_params;
  if (!problem.ParseFromArray(problem_str, problem_str_len))
    return nullptr;
  if (!solver_params.ParseFromArray(solver_params_str, solver_params_str_len))
    return nullptr;

  DataMap data_map;
  WriteConstants(data, &data_map);
  Solver* solver;
  std::unique_ptr<Solver> solver_gc;

  // In warm start mode, cache the solver object
  if (solver_params.warm_start()) {
    auto iter = global_solver_cache.find(problem_str);
    if (iter == global_solver_cache.end()) {
      auto retval = global_solver_cache.insert(
          make_pair(
              problem_str,
              CreateSolver(problem, data_map, solver_params)));
      iter = retval.first;
    }
    solver = iter->second.get();
  } else {
    solver_gc = CreateSolver(problem, data_map, solver_params);
    solver = solver_gc.get();
  }
  SetParameterValues(parameters, solver);

  if (!setjmp(failure_buf)) {
    // Standard execution path
    BlockVector block_x = solver->Solve();
    std::string status_str = solver->status().SerializeAsString();

    // Get results
    PyObject* vars = PyDict_New();
    {
      for (const Expression* expr : GetVariables(problem)) {
        const std::string& var_id = expr->variable().variable_id();
        Eigen::VectorXd x = block_x(var_id);

        PyObject* val = PyString_FromStringAndSize(
            reinterpret_cast<const char*>(x.data()),
            x.rows()*sizeof(double));

        PyDict_SetItemString(vars, var_id.c_str(), val);
        Py_DECREF(val);
      }
    }

    PyObject* retval = Py_BuildValue("s#O", status_str.data(), status_str.size(), vars);
    Py_DECREF(vars);
    return retval;
  }

  // Error execution path
  PyErr_SetString(SolveError, "CHECK failed");
  return nullptr;
}

static PyObject* EvalProx(PyObject* self, PyObject* args) {
  const char* f_expr_str;
  int f_expr_str_len;
  double lambda;
  PyObject* data;
  PyObject* v_map;

  // prox(expr_str, lambda, data, v_map)
  if (!PyArg_ParseTuple(
          args, "s#dOO",
          &f_expr_str, &f_expr_str_len, &lambda, &data, &v_map)) {
    // TODO(mwytock): Need to set the appropriate exceptions when passed
    // incorrect arguments.
    return nullptr;
  }

  Expression f_expr;
  if (!f_expr.ParseFromArray(f_expr_str, f_expr_str_len))
    return nullptr;

  DataMap data_map;
  WriteConstants(data, &data_map);
  if (!setjmp(failure_buf)) {
    CHECK_EQ(Expression::PROX_FUNCTION, f_expr.expression_type());

    AffineOperator H, A;
    for (int i = 0; i < f_expr.arg_size(); i++) {
      affine::BuildAffineOperator(
          f_expr.arg(i), data_map, affine::arg_key(i), &H.A, &H.b);
    }

    // Set up affine function for constraints for (1/2)||A(x) - v||^2 form
    int i = 0;
    for (const Expression* var_expr : GetVariables(f_expr)) {
      const std::string& var_id = var_expr->variable().variable_id();
      A.A(affine::constraint_key(i++), var_id) = (
          (1/sqrt(lambda))*linear_map::Identity(GetDimension(*var_expr)));
    }
    BlockVector v = A.A*GetVariableVector(v_map);

    std::unique_ptr<ProxOperator> op = CreateProxOperator(
        f_expr.prox_function().prox_function_type(),
        f_expr.prox_function().epigraph());
    op->Init(ProxOperatorArg(f_expr.prox_function(), data_map, H, A));
    BlockVector x = op->Apply(v);
    PyObject* vars = GetVariableMap(x);
    PyObject* retval = Py_BuildValue("O", vars);
    Py_DECREF(vars);
    return retval;
  }

  PyErr_SetString(SolveError, "CHECK failed");
  return nullptr;
}


void HandleFailure() {
  // TODO(mwytock): Dump stack trace here
  longjmp(failure_buf, 1);
}

void LogVerbose_PySys(const std::string& msg) {
  PySys_WriteStdout("%s\n", msg.c_str());
}

void InitLogging() {
  google::InitGoogleLogging("epopt");
  // const char* v = getenv("EPSILON_VLOG");
  // if (v != nullptr)
  //   FLAGS_v = atoi(v);

  google::InstallFailureFunction(&HandleFailure);
  SetVerboseLogger(&LogVerbose_PySys);
  // TODO(mwytock): Should we set up glog so that VLOG uses PySys_WriteStderr?
}

static PyMethodDef SolveMethods[] = {
  {"solve", Solve, METH_VARARGS,
   "Solve a problem with epsilon."},
  {"eval_prox", EvalProx, METH_VARARGS,
   "Test a proximal operator."},
  {nullptr, nullptr, 0, nullptr}
};


static bool initialized = false;
PyMODINIT_FUNC init_solve() {
  // TODO(mwytock): Increase logging verbosity based on environment variable
  if (!initialized) {
    InitLogging();
    initialized = true;
  }

  PyObject* m = Py_InitModule("_solve", SolveMethods);
  if (m == nullptr)
    return;

  SolveError = PyErr_NewException(
      const_cast<char*>("_solve.error"), nullptr, nullptr);
  Py_INCREF(SolveError);
  PyModule_AddObject(m, "error", SolveError);
}

}  // extern "C"
