"""Standard form QP."""

import cvxpy as cp
import numpy as np
import numpy.linalg as LA

def create(n):
    np.random.seed(0)
    
    P = np.random.rand(n,n);
    P = P.T.dot(P) + np.eye(n)
    q = np.random.randn(n);
    r = np.random.randn();

    l = np.random.randn(n);
    u = np.random.randn(n);
    lb = np.minimum(l,u);
    ub = np.maximum(l,u);

    x = cp.Variable(n)
    f = 0.5*cp.quad_form(x, P) + q.T*x + r
    C = [x >= lb,
         x <= ub]
    return cp.Problem(cp.Minimize(f), C)
    
