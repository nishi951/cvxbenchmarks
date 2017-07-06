"""Functional form of the expression operators."""

import numpy as np

from epopt import affine
from epopt import constant as _constant
from epopt import dcp
from epopt.error import ExpressionError
from epopt.expression_util import *
from epopt.proto.epsilon import expression_pb2
from epopt.proto.epsilon.expression_pb2 import Monotonicity, Curvature, Sign, Size, Cone
from functools import reduce

# Shorthand convenience
SIGNED = Monotonicity(monotonicity_type=Monotonicity.SIGNED)
INCREASING = Monotonicity(monotonicity_type=Monotonicity.INCREASING)

AFFINE = Curvature(curvature_type=Curvature.AFFINE)
CONSTANT = Curvature(curvature_type=Curvature.CONSTANT)

# Thin wrappers around Expression/Problem protobuf, making them immutable and
# with reference semantics for args.
class Problem(object):
    def __init__(self, objective, constraint=[], data={}):
        assert type(objective) is Expression
        for constr in constraint:
            assert type(constr) is Expression

        self.objective = objective
        self.constraint = constraint
        self.data = data

    def SerializeToString(self):
        proto = expression_pb2.Problem(
            objective=self.objective.proto_with_args,
            constraint=[c.proto_with_args for c in self.constraint])
        return proto.SerializeToString()

    def expression_data(self):
        retval = self.objective.expression_data()
        for constr in self.constraint:
            retval.update(constr.expression_data())
        return retval

ARG_FIELD = "arg"
DATA_FIELD = "data"
class Expression(object):
    def __init__(self, **kwargs):
        self.arg = list(kwargs.get(ARG_FIELD, []))
        self.data = kwargs.get(DATA_FIELD, {})
        for arg in self.arg:
            assert type(arg) is Expression

        kwargs[ARG_FIELD] = []
        if DATA_FIELD in kwargs:
            del kwargs[DATA_FIELD]
        self.proto = expression_pb2.Expression(**kwargs)

        # Lazily computed properties
        self._dcp_props = None
        self._affine_props = None

    def __eq__(self, other):
        if isinstance(other, Expression):
            return self.proto_with_args == other.proto_with_args
        else:
            return False

    @property
    def dcp_props(self):
        if self._dcp_props is None:
            self._dcp_props = dcp.compute_dcp_properties(self)
        return self._dcp_props

    @property
    def affine_props(self):
        if self._affine_props is None:
            self._affine_props = affine.compute_affine_properties(self)
        return self._affine_props

    @property
    def proto_with_args(self):
        proto_with_args = expression_pb2.Expression()
        proto_with_args.CopyFrom(self.proto)
        proto_with_args.arg.extend(arg.proto_with_args for arg in self.arg)
        return proto_with_args

    def SerializeToString(self):
        return self.proto_with_args.SerializeToString()

    def __getattr__(self, name):
        return getattr(self.proto, name)

    def expression_data(self):
        retval = dict(self.data)
        for arg in self.arg:
            retval.update(arg.expression_data())
        return retval

def from_proto(proto, arg, data):
    assert not proto.arg
    expr = Expression()
    expr.proto = proto
    expr.arg = arg
    expr.data = data
    return expr

def is_scalar(a):
    return a[0]*a[1] == 1

def elementwise_dims(a, b):
    if a == b:
        return a
    if is_scalar(b):
        return a
    if is_scalar(a):
        return b
    raise ValueError("Incompatible elemwise binary op sizes")

def matrix_multiply_dims(a, b):
    if a[1] == b[0]:
        return (a[0], b[1])
    if is_scalar(a):
        return b
    if is_scalar(b):
        return a
    raise ValueError("Incompatible matrix multiply sizes")

def stack_dims(a, b, idx):
    if a[1-idx] != b[1-idx]:
        raise ValueError("Incomaptible stack sizes")
    new_dims = list(a)
    new_dims[idx] += b[idx]
    return tuple(new_dims)

def _multiply(args, elemwise=False):
    if not args:
        raise ValueError("multiplying null args")

    op_dims = elementwise_dims if elemwise else matrix_multiply_dims
    return Expression(
        expression_type=(expression_pb2.Expression.MULTIPLY_ELEMENTWISE if elemwise else
                         expression_pb2.Expression.MULTIPLY),
        arg=args,
        size=Size(dim=reduce(lambda a, b: op_dims(a, b),
                             (dims(a) for a in args))),
        func_curvature=AFFINE)

# Expressions
def add(*args):
    if not args:
        raise ValueError("adding null args")

    return Expression(
        expression_type=expression_pb2.Expression.ADD,
        arg=args,
        size=Size(
            dim=reduce(lambda a, b: elementwise_dims(a, b),
                       (dims(a) for a in args))),
        arg_monotonicity=len(args)*[INCREASING],
        func_curvature=AFFINE)

def multiply(*args):
    return _multiply(args, elemwise=False)

def multiply_elemwise(*args):
    return _multiply(args, elemwise=True)

def hstack(*args):
    return Expression(
        expression_type=expression_pb2.Expression.HSTACK,
        func_curvature=AFFINE,
        size=Size(
            dim=reduce(lambda a, b: stack_dims(a, b, 1),
                       (dims(a) for a in args))),
        arg=args)

def vstack(*args):
    return Expression(
        expression_type=expression_pb2.Expression.VSTACK,
        func_curvature=AFFINE,
        size=Size(
            dim=reduce(lambda a, b: stack_dims(a, b, 0),
                       (dims(a) for a in args))),
        arg=args)

def reshape(arg, m, n):
    if dim(arg, 0) == m and dim(arg, 1) == n:
        return arg

    if m*n != dim(arg):
        raise ExpressionError("cant reshape to %d x %d" % (m, n), arg)

    # If we have two reshapes that "undo" each other, cancel them out
    if (arg.expression_type == expression_pb2.Expression.RESHAPE and
        dim(arg.arg[0], 0) == m and
        dim(arg.arg[0], 1) == n):
        return arg.arg[0]

    return Expression(
        expression_type=expression_pb2.Expression.RESHAPE,
        arg=[arg],
        size=Size(dim=[m,n]),
        func_curvature=AFFINE,
        sign=arg.sign)

def negate(x):
    # Automatically reduce negate(negate(x)) to x
    if x.expression_type == expression_pb2.Expression.NEGATE:
        return only_arg(x)

    return Expression(
        expression_type=expression_pb2.Expression.NEGATE,
        arg=[x],
        size=x.size,
        func_curvature=AFFINE)

def variable(m, n, variable_id):
    return Expression(
        expression_type=expression_pb2.Expression.VARIABLE,
        size=Size(dim=[m, n]),
        variable=expression_pb2.Variable(variable_id=variable_id),
        func_curvature=Curvature(
            curvature_type=Curvature.AFFINE,
            elementwise=True,
            scalar_multiple=True))

def parameter(m, n, parameter_id, constant_type, sign):
    # NOTE(mwytock): we assume all parameters are dense matrices for purposes of
    # symbolic transformation.
    return Expression(
        expression_type=expression_pb2.Expression.CONSTANT,
        size=Size(dim=[m, n]),
        func_curvature=CONSTANT,
        constant=expression_pb2.Constant(
            constant_type=constant_type, parameter_id=parameter_id, m=m, n=n),
        sign=sign)

def scalar_constant(scalar, size=None):
    if size is None:
        size = (1, 1)

    return Expression(
        expression_type=expression_pb2.Expression.CONSTANT,
        size=Size(dim=size),
        constant=expression_pb2.Constant(
            constant_type=expression_pb2.Constant.SCALAR,
            scalar=scalar),
        func_curvature=CONSTANT)

def ones(*dims):
    data = {}
    return Expression(
        expression_type=expression_pb2.Expression.CONSTANT,
        size=Size(dim=dims),
        data=data,
        constant=_constant.store(np.ones(dims), data),
        func_curvature=CONSTANT)

def constant(m, n, scalar=None, constant=None, sign=None, data={}):
    if scalar is not None:
        constant = expression_pb2.Constant(
            constant_type=expression_pb2.Constant.SCALAR,
            scalar=scalar)
        if scalar > 0:
            sign = Sign(sign_type=expression_pb2.Sign.POSITIVE)
        elif scalar < 0:
            sign = Sign(sign_type=expression_pb2.Sign.NEGATIVE)
        else:
            sign = Sign(sign_type=expression_pb2.Sign.ZERO)

    elif constant is None:
        raise ValueError("need either scalar or constant")

    return Expression(
        expression_type=expression_pb2.Expression.CONSTANT,
        data=data,
        size=Size(dim=[m, n]),
        constant=constant,
        func_curvature=Curvature(curvature_type=Curvature.CONSTANT),
        sign=sign)

def indicator(cone_type, *args):
    return Expression(
        expression_type=expression_pb2.Expression.INDICATOR,
        size=Size(dim=[1, 1]),
        cone=Cone(cone_type=cone_type),
        arg=args)

def norm_pq(x, p, q):
    return Expression(
        expression_type=expression_pb2.Expression.NORM_PQ,
        size=Size(dim=[1, 1]),
        arg=[x], p=p, q=q)

def norm_p(x, p):
    return Expression(
        expression_type=expression_pb2.Expression.NORM_P,
        size=Size(dim=[1, 1]),
        arg=[x], p=p)

def power(x, p):
    return Expression(
        expression_type=expression_pb2.Expression.POWER,
        size=x.size,
        arg=[x], p=p)

def sum_largest(x, k):
    return Expression(
        expression_type=expression_pb2.Expression.SUM_LARGEST,
        size=Size(dim=[1,1]),
        arg=[x], k=k)

def abs_val(x):
    return Expression(
        expression_type=expression_pb2.Expression.ABS,
        arg_monotonicity=[SIGNED],
        size=x.size,
        arg=[x])

def sum_entries(x):
    return Expression(
        expression_type=expression_pb2.Expression.SUM,
        size=Size(dim=[1, 1]),
        func_curvature=AFFINE,
        arg=[x])

def exp(x):
    return Expression(
        expression_type=expression_pb2.Expression.EXP,
        arg_monotonicity=[INCREASING],
        size=x.size,
        arg=[x])

def transpose(x):
    m, n = x.size.dim
    return Expression(
        expression_type=expression_pb2.Expression.TRANSPOSE,
        size=Size(dim=[n, m]),
        func_curvature=AFFINE,
        arg=[x])

def trace(X):
    return Expression(
        expression_type=expression_pb2.Expression.TRACE,
        size=Size(dim=[1, 1]),
        func_curvature=AFFINE,
        arg=[X])

def diag_vec(x):
    if dim(x, 1) != 1:
        raise ExpressionError("diag_vec on non vector")

    n = dim(x, 0)
    return Expression(
        expression_type=expression_pb2.Expression.DIAG_VEC,
        size=Size(dim=[n, n]),
        func_curvature=AFFINE,
        arg=[x])

def index(x, start_i, stop_i, start_j=None, stop_j=None):
    if start_j is None and stop_j is None:
        start_j = 0
        stop_j = x.size.dim[1]

    if (dim(x, 0) == stop_i - start_i and
        dim(x, 1) == stop_j - start_j):
        return x

    return Expression(
        expression_type=expression_pb2.Expression.INDEX,
        size=Size(dim=[stop_i-start_i, stop_j-start_j]),
        func_curvature=AFFINE,
        key=[expression_pb2.Slice(start=start_i, stop=stop_i, step=1),
             expression_pb2.Slice(start=start_j, stop=stop_j, step=1)],
        arg=[x])

def zero(x):
    return Expression(
        expression_type=expression_pb2.Expression.ZERO,
        size=Size(dim=[1, 1]),
        arg=[x])

def linear_map(A, x):
    if dim(x, 1) != 1:
        raise ExpressionError("applying linear map to non vector", x)
    if A.n != dim(x):
        raise ExpressionError("linear map has wrong size: %s" % A, x)

    return Expression(
        expression_type=expression_pb2.Expression.LINEAR_MAP,
        size=Size(dim=[A.m, 1]),
        func_curvature=AFFINE,
        linear_map=A.proto,
        data=A.data,
        arg=[x])

def eq_constraint(x, y):
    if dims(x) != dims(y) and dim(x) != 1 and dim(y) != 1:
        raise ExpressionError("incompatible sizes", x, y)
    return indicator(Cone.ZERO, add(x, negate(y)))

def leq_constraint(a, b):
    return indicator(Cone.NON_NEGATIVE, add(b, negate(a)))

# Second order cone constraints are arranged over the rows of x
def soc_constraint(t, x):
    if dim(t, 1) != 1 or dim(t, 0) != dim(x, 0):
        raise ExpressionError("Second order cone, invalid dimensions", t)
    return indicator(Cone.SECOND_ORDER, t, x)

def soc_elemwise_constraint(t, *args):
    t = reshape(t, dim(t), 1)
    X = hstack(*(reshape(arg, dim(arg), 1) for arg in args))
    if dim(t) != dim(X, 0):
        raise ExpressionError("Second order cone, incompatible sizes", t, X)
    return indicator(Cone.SECOND_ORDER, t, X)

def psd_constraint(X, Y):
    return indicator(Cone.SEMIDEFINITE, add(X, negate(Y)))

def semidefinite(X):
    return indicator(Cone.SEMIDEFINITE, X)

def non_negative(x):
    return indicator(Cone.NON_NEGATIVE, x)

def prox_function(f, *args, **kwargs):
    return Expression(
        expression_type=expression_pb2.Expression.PROX_FUNCTION,
        data=kwargs.get("data", {}),
        size=Size(dim=kwargs.get("size", (1,1))),
        prox_function=f,
        arg=args)
