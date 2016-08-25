
from collections import defaultdict
import struct
import random

from cvxpy.utilities import power_tools

from epopt import error
from epopt import expression
from epopt.expression_util import *
from epopt.proto.epsilon.expression_pb2 import Expression, Curvature, Cone, ProxFunction
from epopt.util import *

class TransformError(error.ExpressionError):
    pass

def epi(f_expr, t_expr):
    """An expression for an epigraph constraint.

    The constraint depends on the curvature of f:
      - f convex,  I(f(x) <= t)
      - f concave, I(f(x) >= t)
      - f affine,  I(f(x) == t)
    """
    f_curvature = f_expr.dcp_props.curvature.curvature_type
    if f_curvature == Curvature.CONVEX:
        return expression.leq_constraint(f_expr, t_expr)
    elif f_curvature == Curvature.CONCAVE:
        return expression.leq_constraint(
            expression.negate(f_expr), expression.negate(t_expr))
    elif f_curvature == Curvature.AFFINE:
        return expression.eq_constraint(f_expr, t_expr);
    raise TransformError(
        "Unknown curvature: %s" % Curvature.Type.Name(f_curvature), f_expr)

def epi_var(expr, name, size=None):
    if size is None:
        size = expr.size.dim
    name += ":%x" % random.getrandbits(32)
    return expression.variable(size[0], size[1], name)

def epi_transform(f_expr, name):
    t_expr = epi_var(f_expr, name)
    epi_f_expr = epi(f_expr, t_expr)
    return t_expr, [epi_f_expr]

# gm()/gm_constrs() translated from cvxpy.utilities.power_tools.gm_constrs()
def gm(t, x, y):
    return expression.soc_elemwise_constraint(
        expression.add(x, y),
        expression.add(x, expression.negate(y)),
        expression.multiply(expression.scalar_constant(2), t))

def gm_constrs(t_expr, x_exprs, p):
    assert power_tools.is_weight(p)
    w = power_tools.dyad_completion(p)
    tree = power_tools.decompose(w)

    # Sigh Python variable scoping..
    gm_vars = [0]
    def create_gm_var():
        var = epi_var(t_expr, "gm_var_%d" % gm_vars[0])
        gm_vars[0] += 1
        return var

    d = defaultdict(create_gm_var)
    d[w] = t_expr
    if len(x_exprs) < len(w):
        x_exprs += [t_expr]

    assert len(x_exprs) == len(w)
    for i, (p, v) in enumerate(zip(w, x_exprs)):
        if p > 0:
            tmp = [0]*len(w)
            tmp[i] = 1
            d[tuple(tmp)] = v

    constraints = []
    for elem, children in tree.items():
        if 1 not in elem:
            constraints += [gm(d[elem], d[children[0]], d[children[1]])]

    return constraints

def get_epigraph(expr):
    if not (expr.expression_type == Expression.INDICATOR and
            expr.cone.cone_type == Cone.NON_NEGATIVE and
            not expr.arg[0].dcp_props.affine and
            expr.arg[0].expression_type == Expression.ADD and
            len(expr.arg[0].arg) == 2):
        return None, None

    exprs = expr.arg[0].arg
    for i in xrange(2):
        if exprs[i].dcp_props.affine:
            t_expr = exprs[i]
            f_expr = expression.negate(exprs[i-1])

    return f_expr, t_expr

def get_scalar_constant(expr):
    if dim(expr) == 1:
        if expr.expression_type == Expression.NEGATE:
            c = get_scalar_constant(expr.arg[0])
            if c is not None:
                return -c
        if (expr.expression_type == Expression.CONSTANT and
            not expr.constant.data_location):
            return expr.constant.scalar

def get_hinge_arg(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.MAX_ELEMENTWISE and
        len(expr.arg[0].arg) == 2):
        if get_scalar_constant(expr.arg[0].arg[0]) == 0:
            return expr.arg[0].arg[1]
        elif get_scalar_constant(expr.arg[0].arg[1]) == 0:
            return expr.arg[0].arg[0]

def get_quantile_arg(expr):
    if (((expr.expression_type == Expression.MULTIPLY and dim(expr.arg[0]) == 1) or
         expr.expression_type == Expression.MULTIPLY_ELEMENTWISE) and
        len(expr.arg) == 2 and
        expr.arg[0].dcp_props.constant):
        return expr.arg[0], expr.arg[1]

    return None, None

def get_total_variation_arg(expr):
    if (expr.expression_type == Expression.NORM_P and expr.p == 1 and
        expr.arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[0].expression_type == Expression.INDEX and
        expr.arg[0].arg[0].arg[0].expression_type == Expression.VARIABLE and
        expr.arg[0].arg[1].expression_type == Expression.NEGATE and
        expr.arg[0].arg[1].arg[0].expression_type == Expression.INDEX and
        expr.arg[0].arg[1].arg[0].arg[0].expression_type ==
        Expression.VARIABLE):

        var_id0 = expr.arg[0].arg[0].arg[0].variable.variable_id
        var_id1 = expr.arg[0].arg[1].arg[0].arg[0].variable.variable_id
        if var_id0 == var_id1:
            return expr.arg[0].arg[0].arg[0]

def is_indicator_prox(prox):
    return prox.epigraph or prox.prox_function_type in (
        ProxFunction.NON_NEGATIVE,
        ProxFunction.SECOND_ORDER_CONE,
        ProxFunction.SEMIDEFINITE,
        ProxFunction.ZERO)
