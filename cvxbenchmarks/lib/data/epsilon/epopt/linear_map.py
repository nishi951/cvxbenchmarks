"""Operations on LinearMaps."""

import numpy as np
import scipy.sparse as sp

from epopt import constant
from epopt.expression_util import *
from epopt.proto.epsilon import expression_pb2

DATA_FIELD = "data"
class LinearMap(object):
    def __init__(self, **kwargs):
        self.data = kwargs.get(DATA_FIELD, {})
        if DATA_FIELD in kwargs:
            del kwargs[DATA_FIELD]
        self.proto = expression_pb2.LinearMap(**kwargs)

    def __getattr__(self, name):
        return getattr(self.proto, name)

# Atomic linear maps
def kronecker_product(A, B):
    if A.m*A.n == 1:
        return B
    if B.m*B.n == 1:
        return A

    if (A.linear_map_type == expression_pb2.LinearMap.SCALAR and
        B.linear_map_type == expression_pb2.LinearMap.SCALAR):
        return scalar(A.scalar*B.scalar, A.n*B.n)

    data = A.data
    data.update(B.data)
    return LinearMap(
        linear_map_type=expression_pb2.LinearMap.KRONECKER_PRODUCT,
        data=data,
        m=A.m*B.m,
        n=A.n*B.n,
        arg=[A.proto, B.proto])

def dense_matrix(constant, data):
    return LinearMap(
        linear_map_type=expression_pb2.LinearMap.DENSE_MATRIX,
        m=constant.m,
        n=constant.n,
        constant=constant,
        data=data)

def sparse_matrix(constant, data):
    return LinearMap(
        linear_map_type=expression_pb2.LinearMap.SPARSE_MATRIX,
        m=constant.m,
        n=constant.n,
        constant=constant,
        data=data)

def diagonal_matrix(constant, data):
    n = constant.m*constant.n
    return LinearMap(
        linear_map_type=expression_pb2.LinearMap.DIAGONAL_MATRIX, m=n, n=n,
        constant=constant,
        data=data)

def scalar(alpha, n):
    return LinearMap(
        linear_map_type=expression_pb2.LinearMap.SCALAR,
        m=n,
        n=n,
        scalar=alpha)

# Operations on linear maps
def transpose(A):
    return LinearMap(
        linear_map_type=expression_pb2.LinearMap.TRANSPOSE,
        data=A.data,
        m=A.n, n=A.m, arg=[A.proto])

# Implementation of various linear maps in terms of atoms
def identity(n):
    return scalar(1, n)

def index(slice, n):
    m = slice.stop - slice.start
    if m == n:
        return identity(n)

    A = sp.coo_matrix(
        (np.ones(m),
         (np.arange(m), np.arange(slice.start, slice.stop, slice.step))),
        shape=(m, n))
    data = {}
    return sparse_matrix(constant.store(A, data), data)

def one_hot(i, n):
    return sparse_matrix(
        constant.store(sp.coo_matrix(([1], ([i], [0])), shape=(n,1))))

def sum(m, n):
    data = {}
    return kronecker_product(
        dense_matrix(constant.store(np.ones((1,n)), data), data),
        dense_matrix(constant.store(np.ones((1,m)), data), data))

def sum_left(m, n):
    data = {}
    return left_matrix_product(dense_matrix(
        constant.store(np.ones((1,m)), data), data), n)

def sum_right(m, n):
    data = {}
    return right_matrix_product(dense_matrix(
        constant.store(np.ones((n,1)), data), data), m)

def promote(n):
    data = {}
    return dense_matrix(constant.store(np.ones((n,1)), data), data)

def negate(n):
    return scalar(-1,n)

def left_matrix_product(A, n):
    return kronecker_product(identity(n), A)

def right_matrix_product(B, m):
    return kronecker_product(transpose(B), identity(m))

def transpose_matrix(m, n):
    A = sp.coo_matrix(
        (np.ones(m*n),
         (np.arange(m*n),
          np.tile(np.arange(n)*m, m) + np.repeat(np.arange(m), n))),
        shape=(m*n, m*n))
    data = {}
    return sparse_matrix(constant.store(A, data), data)

# NOTE(mwytock): Represent the following functions as sparse matrices. This is
# not very efficient, but we expect these to be relatively rare so the sparse
# matrix form should be fine.
def diag_mat(n):
    data = {}
    rows = np.arange(n)
    cols = np.arange(n)*(n+1)
    A = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(n, n*n))
    return sparse_matrix(constant.store(A, data), data)

def diag_vec(n):
    data = {}
    rows = np.arange(n)*(n+1)
    cols = np.arange(n)
    A = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(n*n, n))
    return sparse_matrix(constant.store(A, data), data)

def trace(n):
    data = {}
    rows = np.zeros(n)
    cols = np.arange(n)*(n+1)
    A = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(1, n*n))
    return sparse_matrix(constant.store(A, data), data)

def upper_tri(n):
    data = {}
    m = n*(n-1)/2
    rows = np.arange(m)
    cols = np.array([j*n + i for i in range(n) for j in range(i+1,n)])
    A = sp.coo_matrix((np.ones(m), (rows, cols)), shape=(m, n*n))
    return sparse_matrix(constant.store(A, data), data)
