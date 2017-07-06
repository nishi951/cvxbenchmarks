"""Convert CVXPY expressions into Expression trees."""

import numpy

from cvxpy.atoms import *

from cvxpy import utilities as u
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.binary_operators import RMulExpression
from cvxpy.atoms.affine.diag import diag_mat, diag_vec
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.elementwise.norm2_elemwise import norm2_elemwise
from cvxpy.constraints.eq_constraint import EqConstraint
from cvxpy.constraints.leq_constraint import LeqConstraint
from cvxpy.constraints.psd_constraint import PSDConstraint
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.soc_elemwise import SOC_Elemwise
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variables.variable import Variable
from cvxpy.problems import objective

from epopt import constant
from epopt import expression
from epopt.proto.epsilon import expression_pb2
from epopt.proto.epsilon.expression_pb2 import Expression

def index_value(index, size):
    if index < 0:
        return size + index
    return index

def variable_id(expr):
    return "var:" + str(expr.id)

def parameter_id(expr):
    return "param:" + str(expr.id)

def convert_variable(expr):
    m, n = expr.size
    return expression.variable(m, n, variable_id(expr))

def convert_parameter(expr):
    m, n = expr.size

    if m*n == 1:
        constant_type = expression_pb2.Constant.SCALAR
    else:
        constant_type = expression_pb2.Constant.DENSE_MATRIX

    sign = expression_pb2.Sign(
        sign_type=expression_pb2.Sign.Type.Value(expr.sign))
    return expression.parameter(m, n, parameter_id(expr), constant_type, sign)

def convert_constant(expr):
    m, n = expr.size
    if isinstance(expr.value, (int, float)):
        return expression.constant(m, n, scalar=expr.value)
    else:
        data = {}
        return expression.constant(
            m, n,
            data=data,
            constant=constant.store(expr.value, data),
            sign=expression_pb2.Sign(
                sign_type=expression_pb2.Sign.Type.Value(expr.sign)))

def convert_generic(expression_type, expr):
    proto = expression.Expression(
        expression_type=expression_type,
        size=expression_pb2.Size(dim=expr.size),
        func_curvature=expression_pb2.Curvature(
            curvature_type=expression_pb2.Curvature.Type.Value(
                expr.func_curvature().curvature_str)),
        sign=expression_pb2.Sign(
            sign_type=expression_pb2.Sign.Type.Value(expr.sign)),
        arg_monotonicity=[
            expression_pb2.Monotonicity(
                monotonicity_type=expression_pb2.Monotonicity.Type.Value(m))
            for m in expr.monotonicity()],
        arg=(convert_expression(arg) for arg in expr.args))

    if isinstance(expr, AxisAtom) and expr.axis is not None:
        proto.proto.has_axis = True
        proto.proto.axis = expr.axis

    return proto

def convert_binary(f, expr):
    return f(*[convert_expression(arg) for arg in expr.args])

def convert_unary(f, expr):
    assert len(expr.args) == 1
    return f(convert_expression(expr.args[0]))

def convert_index(expr):
    starts = []
    stops = []
    assert len(expr.key) == 2
    for i, key in enumerate(expr.key):
        size = expr.args[0].size[i]
        starts.append(index_value(key.start, size) if key.start else 0)
        stops.append(index_value(key.stop, size) if key.stop else size)

    assert len(expr.args) == 1
    return expression.index(convert_expression(expr.args[0]),
                            starts[0], stops[0],
                            starts[1], stops[1])

def convert_huber(expr):
    proto = convert_generic(Expression.HUBER, expr)
    proto.proto.M = expr.M.value
    return proto

def convert_p(expression_type, expr):
    proto = convert_generic(expression_type, expr)
    proto.proto.p = float(expr.p)
    return proto

def convert_fraction(fraction):
    return expression_pb2.Fraction(
        a=fraction.numerator,
        b=fraction.denominator)

def convert_geo_mean(expr):
    proto = convert_generic(Expression.GEO_MEAN, expr)
    proto.proto.geo_mean_params.w.extend(convert_fraction(x) for x in expr.w)
    proto.proto.geo_mean_params.w_dyad.extend(convert_fraction(x) for x in expr.w_dyad)
    return proto

def convert_sum_largest(expr):
    proto = convert_generic(Expression.SUM_LARGEST, expr)
    proto.proto.k = expr.k
    return proto

EXPRESSION_TYPES = (
    (AddExpression, lambda e: convert_binary(expression.add, e)),
    (Constant, convert_constant),
    (MulExpression, lambda e: convert_binary(expression.multiply, e)),
    (RMulExpression, lambda e: convert_binary(expression.multiply, e)),
    (NegExpression, lambda e: convert_unary(expression.negate, e)),
    (Parameter, convert_parameter),
    (Variable, convert_variable),
    (abs, lambda e: convert_generic(Expression.ABS, e)),
    (diag_mat, lambda e: convert_generic(Expression.DIAG_MAT, e)),
    (diag_vec, lambda e: convert_generic(Expression.DIAG_VEC, e)),
    (entr, lambda e: convert_generic(Expression.ENTR, e)),
    (exp, lambda e: convert_generic(Expression.EXP, e)),
    (geo_mean, convert_geo_mean),
    (hstack, lambda e: convert_generic(Expression.HSTACK, e)),
    (huber, convert_huber),
    (index, convert_index),
    (kl_div, lambda e: convert_generic(Expression.KL_DIV, e)),
    (kron, lambda e: convert_generic(Expression.KRON, e)),
    (lambda_max, lambda e: convert_generic(Expression.LAMBDA_MAX, e)),
    (log, lambda e: convert_generic(Expression.LOG, e)),
    (log_det, lambda e: convert_generic(Expression.LOG_DET, e)),
    (log_sum_exp, lambda e: convert_generic(Expression.LOG_SUM_EXP, e)),
    (logistic, lambda e: convert_generic(Expression.LOGISTIC, e)),
    (matrix_frac, lambda e: convert_generic(Expression.MATRIX_FRAC, e)),
    (max_elemwise, lambda e: convert_generic(Expression.MAX_ELEMENTWISE, e)),
    (max_entries, lambda e: convert_generic(Expression.MAX_ENTRIES, e)),
    (min_elemwise, lambda e: convert_generic(Expression.MIN_ELEMENTWISE, e)),
    (mul_elemwise, lambda e: convert_binary(expression.multiply_elemwise, e)),
    (norm2_elemwise, lambda e: convert_generic(Expression.NORM_2_ELEMENTWISE, e)),
    (normNuc, lambda e: convert_generic(Expression.NORM_NUC, e)),
    (pnorm, lambda e: convert_p(Expression.NORM_P, e)),
    (power, lambda e: convert_p(Expression.POWER, e)),
    (quad_over_lin, lambda e: convert_generic(Expression.QUAD_OVER_LIN, e)),
    (reshape, lambda e: convert_generic(Expression.RESHAPE, e)),
    (sigma_max, lambda e: convert_generic(Expression.SIGMA_MAX, e)),
    (sum_entries, lambda e: convert_generic(Expression.SUM, e)),
    (sum_largest, lambda e: convert_sum_largest(e)),
    (trace, lambda e: convert_generic(Expression.TRACE, e)),
    (transpose, lambda e: convert_unary(expression.transpose, e)),
    (upper_tri, lambda e: convert_generic(Expression.UPPER_TRI, e)),
    (vstack, lambda e: convert_generic(Expression.VSTACK, e)),
)

# Sanity check to make sure the CVXPY atoms are all classes. This can change
# periodically due to implementation details of CVXPY.
import inspect
for expr_cls, expr_type in EXPRESSION_TYPES:
    assert inspect.isclass(expr_cls), expr_cls

def convert_expression(expr):
    if expr.is_constant():
        return convert_constant(expr)
    else:
        for expr_cls, convert in EXPRESSION_TYPES:
            if type(expr) is expr_cls:
                return convert(expr)
    raise RuntimeError("Unknown type: %s" % type(expr))

def convert_constraint(constraint):
    if isinstance(constraint, EqConstraint):
        return expression.eq_constraint(
            convert_expression(constraint.args[0]),
            convert_expression(constraint.args[1]))
    elif isinstance(constraint, PSDConstraint):
        return expression.psd_constraint(
            convert_expression(constraint.args[0]),
            convert_expression(constraint.args[1]))
    elif isinstance(constraint, LeqConstraint):
        return expression.leq_constraint(
            convert_expression(constraint.args[0]),
            convert_expression(constraint.args[1]))
    elif isinstance(constraint, SOC_Elemwise):
        return expression.soc_elemwise_constraint(
            convert_expression(constraint.t),
            *[convert_expression(x) for x in constraint.x_elems])
    elif isinstance(constraint, SOC):
        return expression.soc_contraint(
            convert_expression(constraint.t),
            expression.vstack(
                *[convert_expression(x) for x in constraint.x_elems]))

    raise RuntimeError("Unknown constraint: %s" % type(constraint))

def convert_problem(problem):
    # NOTE(mwytock): Maximize inherits from Minimize so this clause first!
    if isinstance(problem.objective, objective.Maximize):
        obj_expr = -problem.objective.args[0]
    elif isinstance(problem.objective, objective.Minimize):
        obj_expr = problem.objective.args[0]
    else:
        raise RuntimeError("Unknown objective: %s" % type(problem.objective))

    return expression.Problem(
        objective=convert_expression(obj_expr),
        constraint=[convert_constraint(c) for c in problem.constraints])
