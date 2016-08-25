# Extracts epsilon problems from their benchmark files
#

# Go to __init__.py in the problems dir and comment out the from epopt.cvxpy_solver import solve.
#
import os.path, sys, inspect

problemsDir = "problems"

# Remove from epopt import _solve line



from epopt.problems.problem_instance import ProblemInstance
from epopt.problems import benchmark


# For
# PROBLEMS = [
#     ProblemInstance("basis_pursuit", basis_pursuit.create, dict(m=1000, n=3000)),
#     ProblemInstance("chebyshev", chebyshev.create, dict(m=100, n=200)),
#     ProblemInstance("covsel", covsel.create, dict(m=100, n=200, lam=0.1)),
#     ProblemInstance("fused_lasso", fused_lasso.create, dict(m=1000, ni=10, k=1000)),
#     ProblemInstance("hinge_l1", hinge_l1.create, dict(m=1500, n=5000, rho=0.01)),
#     ProblemInstance("hinge_l1_sparse", hinge_l1.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
#     ProblemInstance("hinge_l2", hinge_l2.create, dict(m=5000, n=1500)),
#     ProblemInstance("hinge_l2_sparse", hinge_l2.create, dict(m=10000, n=1500, mu=0.1)),
#     ProblemInstance("huber", huber.create, dict(m=5000, n=200)),
#     ProblemInstance("infinite_push", infinite_push.create, dict(m=100, n=200, d=20)),
#     ProblemInstance("lasso", lasso.create, dict(m=1500, n=5000, rho=0.01)),
#     ProblemInstance("lasso_sparse", lasso.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
#     ProblemInstance("least_abs_dev", least_abs_dev.create, dict(m=5000, n=200)),
#     ProblemInstance("logreg_l1", logreg_l1.create, dict(m=1500, n=5000, rho=0.01)),
#     ProblemInstance("logreg_l1_sparse", logreg_l1.create, dict(m=1500, n=50000, rho=0.01, mu=0.1)),
#     ProblemInstance("lp", lp.create, dict(m=800, n=1000)),
#     ProblemInstance("max_gaussian", max_gaussian.create, dict(m=10, n=10, k=3)),
#     ProblemInstance("max_softmax", max_softmax.create, dict(m=100, k=20, n=50, epsilon_eps=1e-3)),
#     ProblemInstance("mnist", mnist.create, dict(data=mnist.DATA_SMALL, n=1000)),
#     ProblemInstance("mv_lasso", lasso.create, dict(m=1500, n=5000, k=10, rho=0.01)),
#     ProblemInstance("oneclass_svm", oneclass_svm.create, dict(m=5000, n=200)),
#     ProblemInstance("portfolio", portfolio.create, dict(m=500, n=500000)),
#     ProblemInstance("qp", qp.create, dict(n=1000)),
#     ProblemInstance("quantile", quantile.create, dict(m=400, n=10, k=100, p=1)),
#     ProblemInstance("robust_pca", robust_pca.create, dict(n=100)),
#     ProblemInstance("robust_svm", robust_svm.create, dict(m=2000, n=600)),
#     ProblemInstance("tv_1d", tv_1d.create, dict(n=100000)),
# ]

