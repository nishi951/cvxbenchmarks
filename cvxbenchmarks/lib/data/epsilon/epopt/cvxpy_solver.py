"""CVXPY-like interfaces for solver."""

import logging
import numpy
import time

from cvxpy.settings import OPTIMAL, OPTIMAL_INACCURATE, SOLVER_ERROR

from epopt import __version__
from epopt import _solve
from epopt import constant
from epopt import cvxpy_expr
from epopt import text_format
from epopt import util
from epopt.compiler import compiler
from epopt.proto.epsilon import solver_params_pb2
from epopt.proto.epsilon import solver_pb2
from epopt.proto.epsilon.solver_pb2 import SolverStatus

problem_cache = {}

EPSILON = "epsilon"

class SolverError(Exception):
    pass

def set_solution(prob, values):
    for var in prob.variables():
        var_id = cvxpy_expr.variable_id(var)
        assert var_id in values
        x = numpy.fromstring(values[var_id], dtype=numpy.double)
        var.value = x.reshape(var.size[1], var.size[0]).transpose()

def cvxpy_status(solver_status):
    if solver_status.state == SolverStatus.OPTIMAL:
        return OPTIMAL
    elif solver_status.state == SolverStatus.MAX_ITERATIONS_REACHED:
        return OPTIMAL_INACCURATE
    return SOLVER_ERROR

def parameter_values(cvxpy_prob, data):
    return [(cvxpy_expr.parameter_id(param),
             constant.store(param.value, data).SerializeToString())
            for param in cvxpy_prob.parameters()]

def compile_problem(cvxpy_prob, solver_params):
    t0 = time.time()
    problem = cvxpy_expr.convert_problem(cvxpy_prob)
    problem = compiler.compile_problem(problem, solver_params)
    t1 = time.time()

    if solver_params.verbose:
        print "Epsilon %s" % __version__
        print "Compiled prox-affine form:"
        print text_format.format_problem(problem),
        print "Epsilon compile time: %.4f seconds" % (t1-t0)
        print
    logging.debug("Compiled prox-affine form:\n%s",
                  text_format.format_problem(problem))
    logging.info("Epsilon compile time: %.4f seconds", t1-t0)

    return problem

def solve(cvxpy_prob, **kwargs):
    # Nothing to do in this case
    if not cvxpy_prob.variables():
        return OPTIMAL, cvxpy_prob.objective.value

    solver_params = solver_params_pb2.SolverParams(**kwargs)
    if solver_params.warm_start:
        problem = problem_cache.get(id(cvxpy_prob))
        if not problem:
            problem = compile_problem(cvxpy_prob, solver_params)
            problem_cache[id(cvxpy_prob)] = problem
    else:
        problem = compile_problem(cvxpy_prob, solver_params)

    t0 = time.time()
    if len(problem.objective.arg) == 1 and not problem.constraint:
        # TODO(mwytock): Should probably parameterize the proximal operators so
        # they can take A=0 instead of just using a large lambda here
        lam = 1e12
        values = _solve.eval_prox(
            problem.objective.arg[0].SerializeToString(),
            lam,
            problem.expression_data(),
            {})
        status = OPTIMAL
    else:
        data = problem.expression_data()
        status_str, values = _solve.solve(
            problem.SerializeToString(),
            parameter_values(cvxpy_prob, data),
            solver_params.SerializeToString(),
            data)
        status = cvxpy_status(SolverStatus.FromString(status_str))
    t1 = time.time()

    logging.info("Epsilon solve time: %.4f seconds", t1-t0)
    if solver_params.verbose:
        print "Epsilon solve time: %.4f seconds" % (t1-t0)

    set_solution(cvxpy_prob, values)
    return status, cvxpy_prob.objective.value

def validate_solver(constraints):
    return True
