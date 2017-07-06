"""Transform a problem to prox-affine form.

TODO(mwytock): Clean up the interaction between expression matching and args
extraction.
"""

import logging

from epopt import expression
from epopt import tree_format
from epopt.compiler.transforms import conic
from epopt.compiler.transforms import linear
from epopt.compiler.transforms.transform_util import *
from epopt.proto.epsilon.expression_pb2 import Cone, Expression, ProxFunction, Problem, Size, Sign

class MatchResult(object):
    def __init__(self, match, prox_expr=None, raw_exprs=[], alpha=1):
        self.match = match
        self.prox_expr = prox_expr
        self.raw_exprs = raw_exprs
        self.alpha = alpha

def convert_diagonal(expr):
    if not expr.dcp_props.affine:
        return epi_transform(expr, "affine")
    linear_expr = linear.transform_expr(expr)
    if linear_expr.affine_props.diagonal:
        return linear_expr, []
    return epi_transform(linear_expr, "diagonal")

def convert_scalar(expr):
    if not expr.dcp_props.affine:
        return epi_transform(expr, "affine")
    linear_expr = linear.transform_expr(expr)
    if linear_expr.affine_props.scalar:
        return linear_expr, []
    return epi_transform(linear_expr, "scalar")

def convert_affine(expr):
    if not expr.dcp_props.affine:
        return epi_transform(expr, "affine")
    return linear.transform_expr(expr), []

def create_prox(**kwargs):
    if "alpha" not in kwargs:
        kwargs["alpha"] = 1
    return ProxFunction(**kwargs)

# Simple functions

def prox_constant(expr):
    if expr.dcp_props.constant:
        return MatchResult(
            True,
            expression.prox_function(
                create_prox(prox_function_type=ProxFunction.CONSTANT),
                linear.transform_expr(expr)))
    else:
        return MatchResult(False)


def prox_affine(expr):
    if expr.dcp_props.affine:
        return MatchResult(
            True,
            expression.prox_function(
                create_prox(prox_function_type=ProxFunction.AFFINE),
                linear.transform_expr(expr)))
    else:
        return MatchResult(False)

# Operators

def prox_add(expr):
    if expr.expression_type == Expression.ADD:
        return MatchResult(True, None, expr.arg)
    return MatchResult(False)

def prox_multiply(expr):
    if expr.expression_type == Expression.MULTIPLY and len(expr.arg) == 2:
        for i, arg in enumerate(expr.arg):
            if dim(arg) == 1 and arg.dcp_props.constant:
                alpha = get_scalar_constant(arg)
                f_expr = expr.arg[1-i]
                break
        else:
            return MatchResult(False)

        return MatchResult(
            True,
            None,
            [f_expr],
            alpha)

    return MatchResult(False)

def prox_negate(expr):
    if expr.expression_type == Expression.NEGATE:
        return MatchResult(
            True,
            None,
            [expr.arg[0]],
            -1)
    return MatchResult(False)

# Elementwise

def prox_norm_1(expr):
    if (expr.expression_type == Expression.NORM_P and
        expr.p == 1):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.NORM_1,
                arg_size=[Size(dim=dims(arg))]),
            diagonal_arg),
        constrs)

def prox_non_negative(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.NON_NEGATIVE and
        expr.arg[0].dcp_props.affine):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.NON_NEGATIVE),
            diagonal_arg),
        constrs)

def prox_sum_deadzone(expr):
    hinge_arg = get_hinge_arg(expr)
    arg = None
    if (hinge_arg and
        hinge_arg.expression_type == Expression.ADD and
        len(hinge_arg.arg) == 2 and
        hinge_arg.arg[0].expression_type == Expression.ABS):
        m = get_scalar_constant(hinge_arg.arg[1])
        if m <= 0:
            arg = hinge_arg.arg[0].arg[0]
    if not arg:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.SUM_DEADZONE,
                scaled_zone_params=ProxFunction.ScaledZoneParams(m=-m),
                arg_size=[Size(dim=dims(arg))]),
            diagonal_arg),
        constrs)

def prox_sum_hinge(expr):
    arg = get_hinge_arg(expr)
    if not arg:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.SUM_HINGE,
                arg_size=[Size(dim=dims(arg))],
                has_axis=expr.has_axis,
                axis=expr.axis),
            diagonal_arg,
            size=dims(expr)),
        constrs)

def prox_sum_quantile(expr):
    arg = None
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.MAX_ELEMENTWISE and
        len(expr.arg[0].arg) == 2):

        alpha, x = get_quantile_arg(expr.arg[0].arg[0])
        beta, y  = get_quantile_arg(expr.arg[0].arg[1])
        if (x is not None and y is not None and x == y):
            if (alpha.sign.sign_type == Sign.NEGATIVE and
                beta.sign.sign_type == Sign.POSITIVE):
                alpha, beta = beta, expression.negate(alpha)
                arg = x
            elif (alpha.sign.sign_type == Sign.POSITIVE and
                  beta.sign.sign_type == Sign.NEGATIVE):
                beta = expression.negate(beta)
                arg = x

    if not arg:
        return MatchResult(False)

    alpha = linear.transform_expr(alpha)
    beta = linear.transform_expr(beta)
    data = alpha.expression_data()
    data.update(beta.expression_data())

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.SUM_QUANTILE,
                arg_size=[Size(dim=dims(arg))],
                scaled_zone_params=ProxFunction.ScaledZoneParams(
                    alpha_expr=alpha.proto_with_args,
                    beta_expr=beta.proto_with_args)),
            diagonal_arg,
            data=data),
        constrs)

def prox_exp(expr):
    if (expr.expression_type == Expression.EXP):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.EXP),
            diagonal_arg),
        constrs)

def prox_sum_exp(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.EXP):
        arg = expr.arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.SUM_EXP),
            diagonal_arg),
        constrs)

def prox_sum_inv_pos(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.POWER and
        expr.arg[0].p == -1):
        arg = expr.arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.SUM_INV_POS),
            diagonal_arg),
        constrs)

def prox_sum_logistic(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.LOGISTIC):
        arg = expr.arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.SUM_LOGISTIC),
            diagonal_arg),
        constrs)

def prox_sum_neg_entr(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.NEGATE and
        expr.arg[0].arg[0].expression_type == Expression.ENTR):
        arg = expr.arg[0].arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.SUM_NEG_ENTR),
            diagonal_arg),
        constrs)

def prox_sum_neg_log(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.NEGATE and
        expr.arg[0].arg[0].expression_type == Expression.LOG):
        arg = expr.arg[0].arg[0].arg[0]
    else:
        return MatchResult(False)

    diagonal_arg, constrs = convert_diagonal(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.SUM_NEG_LOG),
            diagonal_arg),
        constrs)

def prox_sum_kl_div(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.KL_DIV):
        args = [expr.arg[0].arg[0], expr.arg[0].arg[1]]
    else:
        return MatchResult(False)

    diagonal_arg0, constrs0 = convert_diagonal(args[0])
    diagonal_arg1, constrs1 = convert_diagonal(args[1])
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.SUM_KL_DIV),
            diagonal_arg0,
            diagonal_arg1),
        constrs0 + constrs1)

# Vector
def prox_log_sum_exp(expr):
    if expr.expression_type == Expression.LOG_SUM_EXP:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.LOG_SUM_EXP,
                arg_size=[Size(dim=dims(arg))],
                has_axis=expr.has_axis,
                axis=expr.axis),
            scalar_arg,
            size=dims(expr)),
        constrs)

def prox_max(expr):
    if expr.expression_type == Expression.MAX_ENTRIES:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.MAX),
            scalar_arg),
        constrs)

def prox_norm_2(expr):
    if expr.expression_type == Expression.NORM_P and expr.p == 2:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.NORM_2),
            scalar_arg),
        constrs)

def prox_sum_largest(expr):
    if expr.expression_type == Expression.SUM_LARGEST:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.SUM_LARGEST,
                sum_largest_params=ProxFunction.SumLargestParams(k=expr.k)),
            scalar_arg),
        constrs)

def prox_total_variation_1d(expr):
    arg = get_total_variation_arg(expr)
    if arg is None:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.TOTAL_VARIATION_1D),
            scalar_arg),
        constrs)

def prox_second_order_cone(expr):
    args = []
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.SECOND_ORDER):
        args = expr.arg
    else:
        f_expr, t_expr = get_epigraph(expr)
        if (f_expr and
            f_expr.expression_type == Expression.NORM_P and
            f_expr.p == 2):
            args = [t_expr, f_expr.arg[0]]
            # make second argument a row vector
            args[1] = expression.reshape(args[1], 1, dim(args[1]))
    if not args:
        return MatchResult(False)

    scalar_arg0, constrs0 = convert_scalar(args[0])
    scalar_arg1, constrs1 = convert_scalar(args[1])
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.SECOND_ORDER_CONE,
                arg_size=[
                    Size(dim=dims(args[0])),
                    Size(dim=dims(args[1]))]),
            scalar_arg0,
            scalar_arg1),
        constrs0 + constrs1)

# Matrix

def prox_lambda_max(expr):
    if expr.expression_type == Expression.LAMBDA_MAX:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.LAMBDA_MAX,
                arg_size=[Size(dim=dims(arg))]),
            scalar_arg),
        constrs)

def prox_log_det(expr):
    if expr.expression_type == Expression.LOG_DET:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                alpha=-1,
                prox_function_type=ProxFunction.NEG_LOG_DET,
                arg_size=[Size(dim=dims(arg))]),
            scalar_arg),
        constrs)

def prox_semidefinite(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.SEMIDEFINITE):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.SEMIDEFINITE,
                arg_size=[Size(dim=dims(arg))]),
            scalar_arg),
        constrs)

def prox_norm_nuclear(expr):
    if expr.expression_type == Expression.NORM_NUC:
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    scalar_arg, constrs = convert_scalar(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(
                prox_function_type=ProxFunction.NORM_NUCLEAR,
                arg_size=[Size(dim=dims(arg))]),
            scalar_arg),
        constrs)

# Any affine function

def prox_sum_square(expr):
    if (expr.expression_type == Expression.QUAD_OVER_LIN and
        expr.arg[1].expression_type == Expression.CONSTANT and
        expr.arg[1].constant.scalar == 1):
        arg = expr.arg[0]
    elif (expr.expression_type == Expression.POWER and
          expr.arg[0].expression_type == Expression.NORM_P and
          expr.p == 2 and expr.arg[0].p == 2):
        arg = expr.arg[0].arg[0]
    else:
        return MatchResult(False)

    affine_arg, constrs = convert_affine(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.SUM_SQUARE),
            affine_arg),
        constrs)

def prox_zero(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.ZERO):
        arg = expr.arg[0]
    else:
        return MatchResult(False)

    affine_arg, constrs = convert_affine(arg)
    return MatchResult(
        True,
        expression.prox_function(
            create_prox(prox_function_type=ProxFunction.ZERO),
            affine_arg),
        constrs)

# Epigraph transform

def epigraph(expr):
    f_expr, t_expr = get_epigraph(expr)
    if f_expr:
        for rule in BASE_RULES:
            result = rule(f_expr)

            if result.match:
                epi_function = result.prox_expr.prox_function
                epi_function.epigraph = True
                epi_function.arg_size.add().CopyFrom(Size(dim=dims(t_expr)))

                linear_t_expr = linear.transform_expr(t_expr)
                if linear_t_expr.affine_props.scalar:
                    constrs = []
                else:
                    linear_t_expr, constrs = epi_transform(
                        linear_t_expr, "scalar")

                return MatchResult(
                    True,
                    expression.prox_function(
                        epi_function, *(result.prox_expr.arg + [linear_t_expr])),
                    result.raw_exprs + constrs)

        # No epigraph transform found, do conic transformation
        obj, constrs = conic.transform_expr(f_expr)
        return MatchResult(
            True,
            None,
            [expression.leq_constraint(obj, t_expr)] + constrs)

    # Not in epigraph form
    return MatchResult(False)

def neg_log_det_epigraph(expr):
    if len(expr.arg[0].arg) != 2:
        return MatchResult(False)

    for i in range(2):
        if expr.arg[0].arg[i].expression_type == Expression.LOG_DET:
            exprs = [expr.arg[0].arg[i],
                        expr.arg[0].arg[1-i]]
            break
    else:
        return MatchResult(False)

    arg = exprs[0].arg[0]
    scalar_arg, constrs = convert_scalar(arg)

    epi_function = create_prox(
                alpha=1,
                prox_function_type=ProxFunction.NEG_LOG_DET,
                arg_size=[Size(dim=dims(arg))])
    epi_function.epigraph = True

    return MatchResult(
        True,
        expression.prox_function(
            epi_function,
            *[scalar_arg, exprs[1]]),
        constrs)

# Conic transform (catch-all default)

def transform_cone(expr):
    obj, constrs = conic.transform_expr(expr)
    return MatchResult(True, None, [obj] + constrs)

# Used for both proximal/epigraph operators
BASE_RULES = [
    # Matrix
    prox_lambda_max,
    prox_log_det,
    prox_norm_nuclear,
    prox_semidefinite,

    # Vector
    prox_log_sum_exp,
    prox_max,
    prox_norm_2,
    prox_second_order_cone,
    prox_sum_largest,
    prox_total_variation_1d,

    # Elementwise
    prox_exp,
    prox_norm_1,
    prox_sum_exp,
    prox_sum_inv_pos,
    prox_sum_logistic,
    prox_sum_neg_entr,
    prox_sum_neg_log,
    prox_sum_kl_div,

    # NOTE(mwytock): Maintain this order as deadzone specializes hinge
    prox_sum_deadzone,
    prox_sum_quantile,
    prox_sum_hinge,

    prox_sum_square,
]

PROX_RULES = [
    # Operators
    prox_add,
    prox_multiply,
    prox_negate,

    # Affine
    prox_zero,

    # Simple
    prox_constant,
    prox_affine,

    # Custom epigraph
    neg_log_det_epigraph,
]

def multiply_scalar(alpha, prox_expr):
    assert prox_expr.expression_type == Expression.PROX_FUNCTION
    if not is_indicator_prox(prox_expr.prox_function):
        prox_expr.prox_function.alpha *= alpha
    return prox_expr

def transform_expr(prox_rules, expr):
    log_debug_expr("prox transform_expr", expr)
    for rule in prox_rules:
        result = rule(expr)

        if result.match:
            logging.debug("match %s", rule.__name__)
            if result.prox_expr:
                yield result.prox_expr

            for raw_expr in result.raw_exprs:
                for prox_expr in transform_expr(prox_rules, raw_expr):
                    yield multiply_scalar(result.alpha, prox_expr)
            break
    else:
        raise TransformError("No rule matched")

def transform_problem(problem, params):
    prox_rules = PROX_RULES + BASE_RULES

    # Epigraph/cone rules
    if params.use_epigraph:
        prox_rules.append(epigraph)
    prox_rules.append(prox_non_negative)
    prox_rules.append(transform_cone)

    f_exprs = list(transform_expr(prox_rules, problem.objective))
    for constr in problem.constraint:
        f_exprs += list(transform_expr(prox_rules, constr))
    return expression.Problem(objective=expression.add(*f_exprs))
