"""MNIST classification task with kitchen sink features."""

import io
import os
import urllib

import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.io
import scipy.sparse as sp

LOCAL_DIR = os.path.dirname(__file__)
DATA_TINY_FILE = os.path.join(LOCAL_DIR, "mnist_tiny.mat")
DATA_SMALL_FILE = os.path.join(LOCAL_DIR, "mnist_small.mat")

DATA_TINY = "file://localhost" + DATA_TINY_FILE    # 20 examples
DATA_SMALL = "file://localhost" + DATA_SMALL_FILE  # 2K examples
DATA_FULL = "http://mnist.mat"                     # 60K examples

def load_data(data_url):
    d = scipy.io.loadmat(io.BytesIO(urllib.urlopen(data_url).read()))
    return d['X'], d['y'].ravel()

def median_dist(X):
    m = X.shape[0]
    k = int(m**1.5)
    I = np.random.randint(0, m, k)
    J = np.random.randint(0, m, k)
    dists = sorted(map(lambda i : LA.norm(X[I[i],:] - X[J[i],:]), xrange(k)))
    return dists[k / 2]

def pca(X):
    dim = 50
    Xc = X - np.mean(X, axis=0)
    _, D = LA.eigh(Xc.T.dot(Xc))
    return X.dot(D[:, :dim])

def random_features(X, n):
    # For small datasets, skip PCA
    Xp = X if X.shape[0] < X.shape[1] else pca(X)
    sigma = median_dist(Xp)
    W = np.random.randn(Xp.shape[1], n) / sigma / np.sqrt(2)
    B = np.random.uniform(0, 2*np.pi, n)
    return np.cos(Xp.dot(W) + B)

def one_hot_encoding(y):
    m = len(y)
    return sp.coo_matrix((np.ones(m), (np.arange(m), y))).toarray()

def create(n, data):
    np.random.seed(0)

    X, y = load_data(data)
    X = random_features(X, n)
    Y = one_hot_encoding(y)
    n = X.shape[1]
    k = Y.shape[1]
    lam = 0.1

    # TODO(mwytock): Use softmax here
    Theta = cp.Variable(n, k)
    f = cp.sum_squares(X*Theta - Y) + lam*cp.norm1(Theta)
    return cp.Problem(cp.Minimize(f))
