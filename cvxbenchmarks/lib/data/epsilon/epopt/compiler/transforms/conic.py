"""Conic transforms for non-linear functions."""

import logging
from fractions import Fraction

from cvxpy.utilities import power_tools

from epopt import expression
from epopt import linear_map
from epopt import tree_format
from epopt.compiler.transforms.transform_util import *
from epopt.expression_util import *
from epopt.proto.epsilon.expression_pb2 import Curvature, Expression

def transform_abs(expr):
    x = only_arg(expr)
    t = epi_var(expr, "abs")
    return t, [expression.leq_constraint(x, t),
               expression.leq_constraint(expression.negate(x), t)]

def transform_max_elementwise(expr):
    t = epi_var(expr, "max_elementwise")
    return t, [expression.leq_constraint(x, t) for x in expr.arg]

def transform_min_elementwise(expr):
    t = epi_var(expr, "min_elementwise")
    return t, [expression.leq_constraint(t, x) for x in expr.arg]

def transform_max_entries(expr):
    x = only_arg(expr)
    m, n = dims(x)
    t = epi_var(expr, "max_entries")
    if not expr.has_axis:
        return t, [expression.leq_constraint(x, t)]
    if expr.axis == 0:
        return t, [
            expression.leq_constraint(
                x, expression.multiply(expression.ones(m, 1), t))]
    if expr.axis == 1:
        return t, [
            expression.leq_constraint(
                x, expression.multiply(t, expression.ones(1, n)))]

    raise TransformError("unknown axis attribute", expr)

def transform_lambda_max(expr):
    t = epi_var(expr, "lambda_max", size=(1,1))
    X = only_arg(expr)
    n = dim(X, 0)
    tI = expression.diag_vec(expression.multiply(expression.ones(n, 1), t))
    return t, [expression.psd_constraint(tI, X)]

def transform_sigma_max(expr):
    X = only_arg(expr)
    m, n = dims(X)
    S = epi_var(expr, "sigma_max_S", size=(m+n, m+n))
    t = epi_var(expr, "sigma_max")
    t_In = expression.diag_vec(expression.multiply(expression.ones(n, 1), t))
    t_Im = expression.diag_vec(expression.multiply(expression.ones(m, 1), t))

    return t, [
        expression.eq_constraint(expression.index(S, 0, n, 0, n), t_In),
        expression.eq_constraint(expression.index(S, n, n+m, 0, n), X),
        expression.eq_constraint(expression.index(S, n, n+m, n, n+m), t_Im),
        expression.semidefinite(S)]

def transform_quad_over_lin(expr):
    assert len(expr.arg) == 2
    x, y = expr.arg
    assert dim(y) == 1

    t = epi_var(expr, "qol", size=(1,1))
    return t, [
        expression.soc_constraint(
            expression.add(y, t),
            expression.hstack(
                expression.add(y, expression.negate(t)),
                expression.reshape(
                    expression.multiply(expression.scalar_constant(2), x),
                    1, dim(x)))),
        expression.leq_constraint(expression.scalar_constant(0), y)]

def transform_norm_p(expr):
    p = expr.p
    x = only_arg(expr)
    t = epi_var(expr, "norm_p")

    if p == float("inf"):
        return t, [expression.leq_constraint(x, t),
                   expression.leq_constraint(expression.negate(x), t)]

    if p == 1:
        return transform_expr(expression.sum_entries(expression.abs_val(x)))

    if p == 2:
        if not expr.has_axis:
            return t, [expression.soc_constraint(
                t,
                expression.reshape(x, 1, dim(x)))]
        if expr.axis == 0:
            return t, [expression.soc_constraint(
                expression.reshape(t, dim(x, 1), 1),
                expression.transpose(x))]
        if expr.axis == 1:
            return t, [expression.soc_constraint(t, x)]

    r = epi_var(expr, "norm_p_r", size=dims(x))
    t1 = expression.multiply(expression.ones(*dims(x)), t)

    if p < 0:
        p, _ = power_tools.pow_neg(p)
        p = Fraction(p)
        constrs = gm_constrs(t1, [x, r], (-p/(1-p), 1/(1-p)))
    elif 0 < p < 1:
        p, _ = power_tools.pow_mid(p)
        p = Fraction(p)
        constrs = gm_constrs(r, [x, t1], (p, 1-p))
    elif p > 1:
        abs_x, constrs = transform_expr(expression.abs_val(x))
        p, _ = power_tools.pow_high(p)
        p = Fraction(p)
        constrs += gm_constrs(abs_x, [r, t1], (1/p, 1-1/p))

    constrs.append(expression.eq_constraint(expression.sum_entries(r), t))
    return t, constrs

def transform_norm_2_elementwise(expr):
    t = epi_var(expr, "norm_2_elementwise")
    return t, [expression.soc_elemwise_constraint(t, *expr.arg)]

def transform_norm_nuc(expr):
    X = only_arg(expr)
    m, n = dims(X)
    T = epi_var(expr, "norm_nuc", size=(m+n, m+n))

    obj = expression.multiply(
        expression.scalar_constant(0.5),
        expression.trace(T))
    return obj, [
        expression.semidefinite(T),
        expression.eq_constraint(expression.index(T, 0, m, m, m+n), X)]

def transform_power(expr):
    p = expr.p

    if p == 1:
        return only_arg(expr)

    one = expression.scalar_constant(1, size=dims(expr))
    if p == 0:
        return one, []

    t = epi_var(expr, "power")
    x = only_arg(expr)

    if p < 0:
        p, w = power_tools.pow_neg(p)
        constrs = gm_constrs(one, [x, t], w)
    if 0 < p < 1:
        p, w = power_tools.pow_mid(p)
        constrs = gm_constrs(t, [x, one], w)
    if p > 1:
        p, w = power_tools.pow_high(p)
        constrs = gm_constrs(x, [t, one], w)

    return t, constrs

def transform_huber(expr):
    n = epi_var(expr, "huber_n")
    s = epi_var(expr, "huber_s")

    # n**2 + 2*M*|s|
    t, constr = transform_expr(
        expression.add(
            expression.power(n, 2),
            expression.multiply(
                expression.scalar_constant(2*expr.M),
                expression.abs_val(s))))
    # x == s + n
    x = only_arg(expr)
    constr.append(expression.eq_constraint(x, expression.add(s, n)))
    return t, constr

def transform_geo_mean(expr):
    w = [Fraction(x.a, x.b) for x in expr.geo_mean_params.w]
    w_dyad = [Fraction(x.a, x.b) for x in expr.geo_mean_params.w_dyad]
    tree = power_tools.decompose(w_dyad)

    t = epi_var(expr, "geo_mean")
    x = only_arg(expr)
    x_list = [expression.index(x, i, i+1) for i in range(len(w))]
    return t, gm_constrs(t, x_list, w)

def transform_sum_largest(expr):
    x = only_arg(expr)
    k = expr.k
    q = epi_var(expr, "sum_largest")
    t = epi_var(expr, "sum_largest_t", size=dims(x))

    obj = expression.add(
        expression.sum_entries(t),
        expression.multiply(expression.scalar_constant(k), q))
    constr = [
        expression.leq_constraint(x, expression.add(t, q)),
        expression.leq_constraint(expression.scalar_constant(0), t)]

    return obj, constr

def transform_matrix_frac(expr):
    assert len(expr.arg) == 2
    x, P = expr.arg
    n = dim(P, 0)

    M = epi_var(expr, "matrix_frac_M", size=(n+1,n+1))
    t = epi_var(expr, "matrix_frac")
    return t, [
        expression.eq_constraint(expression.index(M, 0, n, 0, n), P),
        expression.eq_constraint(expression.index(M, 0, n, n, n+1), x),
        expression.eq_constraint(expression.index(M, n, n+1, n, n+1), t),
        expression.semidefinite(M)]

def transform_exp(expr):
    x = only_arg(expr)
    t = epi_var(expr, "exp")
    return t, [expression.leq_constraint(expr, t)]

def transform_log(expr):
    x = only_arg(expr)
    t = epi_var(expr, "log")
    return t, [expression.leq_constraint(expression.exp(t), x)]

def transform_indicator(expr):
    return expression.scalar_constant(0, size=dims(expr)), [expr]

def transform_expr(expr):
    log_debug_expr("conic transform_expr", expr)

    constrs = []
    transformed_args = []
    for arg in expr.arg:
        obj_arg, constr = transform_expr(arg)
        transformed_args.append(obj_arg)
        constrs += constr

    # Create the same expression but now with linear arguments.
    obj_linear = expression.from_proto(expr.proto, transformed_args, expr.data)

    if not obj_linear.dcp_props.affine:
        f_name = ("transform_" +
                  Expression.Type.Name(obj_linear.expression_type).lower())
        if f_name not in globals():
            raise TransformError("No conic transform", expr)
        obj_linear, constr = globals()[f_name](obj_linear)
        constrs += constr

    return obj_linear, constrs
