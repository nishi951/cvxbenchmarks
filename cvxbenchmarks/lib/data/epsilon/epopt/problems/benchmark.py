#!/usr/bin/env python

import argparse
import logging
import sys
import time

import cvxpy as cp
import numpy as np

from epopt import cvxpy_expr
from epopt import cvxpy_solver
from epopt.compiler import compiler
from epopt.problems import *
from epopt.problems import benchmark_util

from epopt.problems.problem_instance import ProblemInstance

# TODO(mwytock): Slow, maybe consider a smaller version?
# ProblemInstance("mv_lasso_sparse", lasso.create, dict(m=1500, n=50000, k=10, rho=0.01, mu=0.1)),

# Block cholesky very slow for this problem
# ProblemInstance("group_lasso", group_lasso.create, dict(m=1500, ni=50, K=200)),


PROBLEMS = [
    ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=1000, n=3000)),
    ProblemInstance("chebyshev", chebyshev.create, dict(m=100, n=200)),
    ProblemInstance("covsel", covsel.create, dict(m=100, n=200, lam=0.1)),
    ProblemInstance("fused_lasso", fused_lasso.create, dict(m=1000, ni=10, k=1000)),
    ProblemInstance("hinge_l1", hinge_l1.create, dict(m=1500, n=5000, rho=0.01)),
    ProblemInstance("hinge_l1_sparse", hinge_l1.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
    ProblemInstance("hinge_l2", hinge_l2.create, dict(m=5000, n=1500)),
    ProblemInstance("hinge_l2_sparse", hinge_l2.create, dict(m=10000, n=1500, mu=0.1)),
    ProblemInstance("huber", huber.create, dict(m=5000, n=200)),
    ProblemInstance("infinite_push", infinite_push.create, dict(m=100, n=200, d=20)),
    ProblemInstance("lasso", lasso.create, dict(m=1500, n=5000, rho=0.01)),
    ProblemInstance("lasso_sparse", lasso.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
    ProblemInstance("least_abs_dev", least_abs_dev.create, dict(m=5000, n=200)),
    ProblemInstance("logreg_l1", logreg_l1.create, dict(m=1500, n=5000, rho=0.01)),
    ProblemInstance("logreg_l1_sparse", logreg_l1.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
    ProblemInstance("lp", lp.create, dict(m=800, n=1000)),
    ProblemInstance("max_gaussian", max_gaussian.create, dict(m=10, n=10, k=3)),
    ProblemInstance("max_softmax", max_softmax.create, dict(m=100, k=20, n=50, epsilon_eps=1e-3)),
    ProblemInstance("mnist", mnist.create, dict(data=mnist.DATA_SMALL, n=1000)),
    ProblemInstance("mv_lasso", lasso.create, dict(m=1500, n=5000, k=10, rho=0.01)),
    ProblemInstance("oneclass_svm", oneclass_svm.create, dict(m=5000, n=200)),
    ProblemInstance("portfolio", portfolio.create, dict(m=500, n=500000)),
    ProblemInstance("qp", qp.create, dict(n=1000)),
    ProblemInstance("quantile", quantile.create, dict(m=400, n=10, k=100, p=1)),
    ProblemInstance("robust_pca", robust_pca.create, dict(n=100)),
    ProblemInstance("robust_svm", robust_svm.create, dict(m=2000, n=600)),
    ProblemInstance("tv_1d", tv_1d.create, dict(n=100000)),
]

# Each problem should take ~1 minute with 2000 iterations using SCS
# ProblemInstance("max_gaussian", max_gaussian.create, dict(m=200, n=100, k=5))
PROBLEMS_ICML = [
    ProblemInstance("chebyshev", chebyshev.create, dict(m=5000, n=200)),
    ProblemInstance("max_softmax", max_softmax.create, dict(m=400, k=120, n=10, epsilon_eps=1e-4, scs_eps=1e-1, ecos_abstol=1e-1)),
    ProblemInstance("oneclass_svm", oneclass_svm.create, dict(m=6000, n=600)),
    ProblemInstance("robust_svm", robust_svm.create, dict(m=2500, n=750)),
]

PROBLEMS_SCALE = []
PROBLEMS_SCALE += [ProblemInstance(
    "lasso_%d" % int(m),
    lasso.create,
    dict(m=int(m), n=10*int(m), rho=1 if m < 50 else 0.01))
    for m in np.logspace(1, np.log10(5000), 20)]
PROBLEMS_SCALE += [ProblemInstance(
    "mv_lasso_%d" % int(m),
    lasso.create,
    dict(m=int(m), n=10*int(m), k=10, rho=1 if m < 50 else 0.01))
    for m in np.logspace(1, np.log10(5000), 20)]
PROBLEMS_SCALE += [ProblemInstance(
    "fused_lasso_%d" % int(m),
    fused_lasso.create,
    dict(m=int(m), ni=10, k=int(m)))
    for m in np.logspace(1, 3, 20)]
PROBLEMS_SCALE += [ProblemInstance(
    "hinge_l2_%d" % int(n),
    hinge_l2.create,
    dict(m=10*int(n), n=int(n)))
    for n in np.logspace(1, np.log10(5000), 20)]
PROBLEMS_SCALE += [ProblemInstance(
    "robust_svm_%d" % int(n),
    robust_svm.create,
    dict(m=3*int(n), n=int(n)))
    for n in np.logspace(1, np.log10(1500), 20)]

PROBLEM_SCALE_ICML = []
PROBLEM_SCALE_ICML += [ProblemInstance(
    "oneclass_svm_%d" % int(n),
    oneclass_svm.create,
    dict(m=10*int(n), n=int(n)))
    for n in np.logspace(1, np.log10(2000), 10)]
PROBLEM_SCALE_ICML += [ProblemInstance(
    "robust_svm_%d" % int(n),
    robust_svm.create,
    dict(m=10*int(n), n=int(n)))
    for n in np.logspace(1, np.log10(2000), 10)]
PROBLEM_SCALE_ICML += [ProblemInstance(
    "chebyshev_%d" % int(n),
    chebyshev.create,
    dict(m=100*int(n), n=100, epsilon_eps=7e-4/n, scs_eps=1e-1))
    for n in np.logspace(1.3, np.log10(200), 10)]
PROBLEM_SCALE_ICML += [ProblemInstance(
    "max_gaussian_%d" % int(n),
    max_gaussian.create,
    dict(m=int(n), n=100, k=5, epsilon_eps=1e-2, scs_eps=1e-2))
    for n in np.logspace(1, np.log10(80), 10)]
PROBLEM_SCALE_ICML += [ProblemInstance(
    "infinite_push_%d" % int(n),
    infinite_push.create,
    dict(m=int(n), n=int(n), d=int(n)))
    for n in np.logspace(1, np.log10(80), 10)]
PROBLEM_SCALE_ICML += [ProblemInstance(
    "max_softmax_%d" % int(n),
    max_softmax.create,
    dict(m=int(n)*10, k=int(n)*3, n=10, scs_eps=9e-3))
    for n in np.logspace(1, np.log10(80), 10)]



def print_constraints(cvxpy_prob):
    for c in cvxpy_prob.constraints:
        print '[CONSTR]', c.__repr__(), c.value, np.linalg.norm(c.violation)

def benchmark_epsilon(cvxpy_prob, **kwargs):
    if args.iterations:
        kwargs["abs_tol"] = 1e-8
        kwargs["rel_tol"] = 1e-8
        kwargs["max_iterations"] = args.iterations
    else:
        kwargs["max_iterations"] = 50000

	if "epsilon_eps" in cvxpy_prob.kwargs:
		kwargs["abs_tol"] = cvxpy_prob.kwargs["epsilon_eps"]
		kwargs["rel_tol"] = cvxpy_prob.kwargs["epsilon_eps"]/100.

    cvxpy_solver.solve(cvxpy_prob, **kwargs)
    if args.debug:
        print_constraints(cvxpy_prob)
    return cvxpy_prob.objective.value

def benchmark_cvxpy(solver, cvxpy_prob):
    kwargs = {"solver": solver,
              "verbose": args.debug}
    if solver == cp.SCS:
        if args.iterations:
            kwargs["use_indirect"] = args.scs_indirect
            kwargs["max_iters"] = args.iterations
            kwargs["eps"] = 1e-8
        else:
            kwargs["max_iters"] = 10000
            kwargs["eps"] = 1e-3
        if "scs_eps" in cvxpy_prob.kwargs:
            kwargs["eps"] = cvxpy_prob.kwargs["scs_eps"]
    else:
        if "ecos_abstol" in cvxpy_prob.kwargs:
            # to prevent ecos breakdown at max_softmax
            kwargs["abstol"] = cvxpy_prob.kwargs["ecos_abstol"]
            kwargs["reltol"] = 1e-1*kwargs["abstol"]
            kwargs["feastol"] = 1e-6*kwargs["abstol"]

    try:
        # TODO(mwytock): ProblemInstanceably need to run this in a separate thread/process
        # and kill after one hour?
        cvxpy_prob.solve(**kwargs)
        if args.debug:
            print_constraints(cvxpy_prob)
        return cvxpy_prob.objective.value
    except cp.error.SolverError:
        # Raised when solver cant handle a problem
        return float("nan")

BENCHMARKS = {
    "epsilon": lambda p: benchmark_epsilon(p),
    "no_epi": lambda p: benchmark_epsilon(p, use_epigraph=False),
    "scs": lambda p: benchmark_cvxpy(cp.SCS, p),
    "ecos": lambda p: benchmark_cvxpy(cp.ECOS, p),
}

def run_benchmarks(benchmarks, problems):
    for problem in problems:
        logging.debug("problem %s", problem.name)

        t0 = time.time()
        np.random.seed(0)
        cvxpy_prob = problem.create()
        f_eval = None
        if isinstance(cvxpy_prob, tuple):
            cvxpy_prob, f_eval = cvxpy_prob
        t1 = time.time()
        logging.debug("creation time %f seconds", t1-t0)

        data = [problem.name]
        for benchmark in benchmarks:
            logging.debug("running %s", benchmark)

            t0 = time.time()
            value = BENCHMARKS[benchmark](cvxpy_prob)
            t1 = time.time()

            if f_eval:
                if args.debug:
                    print "Use corrected objective"
                value = f_eval()

            logging.debug("done %f seconds", t1-t0)
            yield benchmark, "%-15s" % problem.name, t1-t0, value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="epsilon")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--list-problems", action="store_true")
    parser.add_argument("--problem")
    parser.add_argument("--problem-match")
    parser.add_argument("--problem-set", default="PROBLEMS")
    parser.add_argument("--scs-indirect", action="store_true")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--write")
    args = parser.parse_args()

    problems = locals()[args.problem_set]
    if args.problem:
        problems = [p for p in problems if p.name == args.problem]
    elif args.problem_match:
        problems = [
            p for p in problems if p.name.startswith(args.problem_match)]

    if args.list_problems:
        for problem in problems:
            print problem.name
        sys.exit(0)

    if args.list_benchmarks:
        for benchmark in BENCHMARKS:
            print benchmark
        sys.exit(0)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.write:
        for problem in problems:
            benchmark_util.write_problem(
                problem.create(), args.write, problem.name)
        sys.exit(0)

    for result in run_benchmarks([args.benchmark], problems):
        print "\t".join(str(x) for x in result)

else:
    args = argparse.Namespace()
