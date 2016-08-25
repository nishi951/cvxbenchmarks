import os

import cvxpy as cp
import numpy as np
import scipy.misc

IMAGE = os.path.join(os.path.dirname(__file__), "baby.jpg")

def create(n, lam):
    # get data
    A = np.rot90(scipy.misc.imread(IMAGE), -1)[400:1400,600:1600]
    Y = scipy.misc.imresize(A, (n,n))

    # set up problem
    X = [cp.Variable(n,n) for i in range(3)]
    f = sum([cp.sum_squares(X[i] - Y[:,:,i]) for i in range(3)]) + lam * cp.tv(*X)
    return cp.Problem(cp.Minimize(f))
