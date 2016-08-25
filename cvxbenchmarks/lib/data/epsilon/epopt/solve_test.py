
import copy
import logging

import cvxpy as cp
import numpy as np

from epopt import cvxpy_solver
from epopt.problems import *
from epopt.problems.problem_instance import ProblemInstance
from epopt.proto.epsilon.solver_params_pb2 import SolverParams

REL_TOL = {
    "hinge_l1": 1e-4,
    "hinge_l1_sparse": 1e-4,
    "oneclass_svm": 1e-4,
    "tv_1d": 1e-4,
}

import logging
logging.basicConfig(level=logging.DEBUG)

# TODO(mwytock): Fix this
#ProblemInstance("group_lasso", group_lasso.create, dict(m=15, ni=5, K=10)),

PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=10, n=30)),
    ProblemInstance("chebyshev", chebyshev.create, dict(m=10, n=20, k=3)),
    ProblemInstance("covsel", covsel.create, dict(m=10, n=20, lam=0.1)),
    ProblemInstance("fused_lasso", fused_lasso.create, dict(m=5, ni=2, k=5, rho=0.5)),
    ProblemInstance("hinge_l1", hinge_l1.create, dict(m=5, n=10, rho=0.1)),
    ProblemInstance("hinge_l1_sparse", hinge_l1.create, dict(m=5, n=20, rho=0.1, mu=0.5)),
    ProblemInstance("hinge_l2", hinge_l2.create, dict(m=20, n=10, rho=1)),
    ProblemInstance("hinge_l2_sparse", hinge_l2.create, dict(m=20, n=10, rho=1, mu=0.5)),
    ProblemInstance("huber", huber.create, dict(m=20, n=10)),
    ProblemInstance("infinite_push", infinite_push.create, dict(m=20, n=20, d=10)),
    ProblemInstance("lasso", lasso.create, dict(m=5, n=20, rho=0.1)),
    ProblemInstance("lasso_sparse", lasso.create, dict(m=5, n=20, rho=0.1, mu=0.5)),
    ProblemInstance("least_abs_dev", least_abs_dev.create, dict(m=10, n=5)),
    ProblemInstance("logreg_l1", logreg_l1.create, dict(m=5, n=10, rho=0.1)),
    ProblemInstance("logreg_l1_sparse", logreg_l1.create, dict(m=5, n=10, rho=0.1, mu=0.5)),
    ProblemInstance("lp", lp.create, dict(m=10, n=20)),
    ProblemInstance("mnist", mnist.create, dict(data=mnist.DATA_TINY, n=10)),
    ProblemInstance("mv_lasso", lasso.create, dict(m=5, n=20, k=2, rho=0.1)),
    ProblemInstance("mv_lasso_sparse", lasso.create, dict(m=5, n=10, k=2, rho=0.1, mu=0.5)),
    ProblemInstance("oneclass_svm", oneclass_svm.create, dict(m=20, n=3)),
    ProblemInstance("portfolio", portfolio.create, dict(m=5, n=10)),
    ProblemInstance("qp", qp.create, dict(n=10)),
    ProblemInstance("quantile", quantile.create, dict(m=40, n=2, k=3)),
    ProblemInstance("robust_pca", robust_pca.create, dict(n=10)),
    ProblemInstance("robust_svm", robust_svm.create, dict(m=50, n=2)),
    ProblemInstance("tv_1d", tv_1d.create, dict(n=10)),
    ProblemInstance("tv_denoise", tv_denoise.create, dict(n=10, lam=1)),
]

PARAMS = [
    dict(use_epigraph=True),
    dict(use_epigraph=False),
    dict(solver="PROX_ADMM_TWO_BLOCK"),
]

def solve_problem(problem_instance, params):
    np.random.seed(0)
    problem = problem_instance.create()
    if isinstance(problem, tuple):
        problem, f_eval = problem
    logging.debug(problem_instance.name)

    problem.solve(solver=cp.SCS)
    obj0 = problem.objective.value

    # per-instance rel_tol
    params["rel_tol"] = REL_TOL.get(problem_instance.name, 1e-3)
    cvxpy_solver.solve(problem, **params)
    obj1 = problem.objective.value

    # A lower objective is okay
    assert obj1 <= obj0 + 1e-2*abs(obj0) + 1e-4, "%.2e vs. %.2e" % (obj1, obj0)

def test_solve():
    for problem in PROBLEMS:
        for params in PARAMS:
            yield solve_problem, problem, dict(params)
