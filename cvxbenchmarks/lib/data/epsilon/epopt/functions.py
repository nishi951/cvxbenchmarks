
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

# Utility functions
def sqdist(X, Y):
    """Squared Euclidean distance for two sets of n-dimensional points."""
    return (
        np.tile(np.sum(X**2, axis=1), (Y.shape[0],1)).T +
        np.tile(np.sum(Y**2, axis=1), (X.shape[0],1)) -
        2*X.dot(Y.T))

# Classification
def hinge_loss(theta, X, y):
    if not all(np.unique(y) == [-1, 1]):
        raise ValueError("y must have binary labels in {-1,1}")
    return cp.sum_entries(cp.max_elemwise(1 - sp.diags([y],[0])*X*theta, 0))

def logistic_loss(theta, X, y):
    if not all(np.unique(y) == [-1, 1]):
        raise ValueError("y must have binary labels in {-1,1}")
    return cp.sum_entries(cp.logistic(-sp.diags([y],[0])*X*theta))

# Multiclass classification
def one_hot(y, k):
    m = len(y)
    return sp.coo_matrix((np.ones(m), (np.arange(m), y)), shape=(m, k)).todense()

def softmax_loss(Theta, X, y):
    k = Theta.size[1]
    Y = one_hot(y, k)
    return (cp.sum_entries(cp.log_sum_exp(X*Theta, axis=1)) -
            cp.sum_entries(cp.mul_elemwise(X.T.dot(Y), Theta)))

def multiclass_hinge_loss(Theta, X, y):
    k = Theta.size[1]
    Y = one_hot(y, k)
    return (cp.sum_entries(cp.max_entries(X*Theta + 1 - Y, axis=1)) -
            cp.sum_entries(cp.mul_elemwise(X.T.dot(Y), Theta)))

# Other probabilistic models
def quantile_loss(alphas, Theta, X, y):
    m, n = X.shape
    k = len(alphas)
    Y = np.tile(y.flatten(), (k, 1)).T
    A = np.tile(alphas, (m, 1))
    Z = X*Theta - Y
    return cp.sum_entries(
        cp.max_elemwise(
            cp.mul_elemwise( -A, Z),
            cp.mul_elemwise(1-A, Z)))


def poisson_loss(theta, X, y):
    return (cp.sum_entries(cp.exp(X*theta)) -
            cp.sum_entries(sp.diags([y],[0])*X*theta))


# Ranking loss
def infinite_push(theta, Xp, Xn):
    m, d = Xp.shape
    n = Xn.shape[0]
    Z = cp.max_elemwise(
        1 - (Xp*theta*np.ones((1,n)) - (Xn*theta*np.ones((1,m))).T), 0)
    return cp.max_entries(cp.sum_entries(Z, axis=0))
