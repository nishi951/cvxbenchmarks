#!/usr/bin/env python

import numpy as np
import cvxpy as cp

from epopt import cvxpy_expr
from epopt import expression_vis
from epopt.compiler import canonicalize

if __name__ == "__main__":
    n = 5
    x = cp.Variable(n)

    # Lasso expression tree
    m = 10
    A = np.random.randn(m,n)
    b = np.random.randn(m)
    lam = 1
    f = cp.sum_squares(A*x - b) + lam*cp.norm1(x)
    prob0 = cvxpy_expr.convert_problem(cp.Problem(cp.Minimize(f)))[0]
    expression_vis.graph(prob0.objective).write("expr_lasso.dot")

    # Canonicalization of a more complicated example
    c = np.random.randn(n)
    f = cp.exp(cp.norm(x) + c.T*x) + cp.norm1(x)
    prob0 = cvxpy_expr.convert_problem(cp.Problem(cp.Minimize(f)))[0]
    expression_vis.graph(prob0.objective).write("expr_epigraph.dot")
    prob1 = canonicalize.transform(prob0)
    expression_vis.graph(prob1.objective).write("expr_epigraph_canon.dot")
