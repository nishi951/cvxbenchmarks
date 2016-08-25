
import struct

from epopt.error import ExpressionError
from epopt.proto.epsilon.expression_pb2 import Curvature, Expression

# Helper functions
def only_arg(expr):
    if len(expr.arg) != 1:
        raise ExpressionError("wrong number of args", expr)
    return expr.arg[0]

def dim(expr, index=None):
    if len(expr.size.dim) != 2:
        raise ExpressioneError("wrong number of dimensions", expr)
    if index is None:
        return expr.size.dim[0]*expr.size.dim[1]
    else:
        return expr.size.dim[index]

def dims(expr):
    return tuple(expr.size.dim)
