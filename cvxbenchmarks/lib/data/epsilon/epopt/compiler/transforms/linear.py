"""Transforms for linear functions."""

import logging

from epopt import error
from epopt import expression
from epopt import linear_map
from epopt import tree_format
from epopt.compiler import validate
from epopt.compiler.transforms.transform_util import *
from epopt.proto.epsilon.expression_pb2 import Problem, Constant
from epopt.util import *

def transform_variable(expr):
    return expression.reshape(expr, dim(expr), 1)

def transform_constant(expr):
    return expression.reshape(expr, dim(expr), 1)

def promote(expr, new_dim):
    if dim(expr) != 1 or dim(expr) == new_dim:
        return expr
    return expression.linear_map(linear_map.promote(new_dim), expr)

def transform_add(expr):
    return expression.add(
        *[promote(transform_expr(e), dim(expr)) for e in expr.arg])

def transform_index(expr):
    return expression.linear_map(
        linear_map.kronecker_product(
            linear_map.index(expr.key[1], dim(only_arg(expr),1)),
            linear_map.index(expr.key[0], dim(only_arg(expr),0))),
        transform_expr(only_arg(expr)))

def multiply_constant(expr, n):
    if expr.expression_type == Expression.CONSTANT:
        if expr.constant.constant_type == Constant.SCALAR:
            return linear_map.scalar(expr.constant.scalar, n)
        if expr.constant.constant_type == Constant.DENSE_MATRIX:
            return linear_map.dense_matrix(expr.constant, expr.data)
        if expr.constant.constant_type == Constant.SPARSE_MATRIX:
            return linear_map.sparse_matrix(expr.constant, expr.data)
    elif expr.expression_type == Expression.TRANSPOSE:
        return linear_map.transpose(multiply_constant(only_arg(expr), n))
    raise TransformError("unknown constant type", expr)

def transform_multiply(expr):
    if len(expr.arg) != 2:
        raise TransformError("wrong number of args", expr)

    m = dim(expr, 0)
    n = dim(expr, 1)
    if expr.arg[0].dcp_props.constant:
        A = multiply_constant(expr.arg[0], m)
        B = promote(transform_expr(expr.arg[1]), n*n)
        return expression.linear_map(linear_map.left_matrix_product(A, n), B)

    if expr.arg[1].dcp_props.constant:
        A = promote(transform_expr(expr.arg[0]), m*m)
        B = multiply_constant(expr.arg[1], n)
        return expression.linear_map(linear_map.right_matrix_product(B, m), A)

    raise TransformError("multiplying non constants", expr)

def transform_kron(expr):
    if len(expr.arg) != 2:
        raise TransformError("Wrong number of arguments", expr)

    if not expr.arg[0].dcp_props.constant:
        raise TransformError("First arg is not constant", expr)

    return expression.linear_map(
        linear_map.kronecker_product_single_arg(
            multiply_constant(expr.arg[0], 1),
            dim(expr.arg[1], 0),
            dim(expr.arg[1], 1)),
        transform_expr(expr.arg[1]))

def multiply_elementwise_constant(expr):
    # TODO(mwytock): Handle this case
    if expr.expression_type != Expression.CONSTANT:
        raise TransformError("multiply constant is not leaf", expr)

    if expr.constant.constant_type == Constant.DENSE_MATRIX:
        return linear_map.diagonal_matrix(expr.constant, expr.data)
    if expr.constant.constant_type == Constant.SCALAR:
        return linear_map.scalar(expr.constant.scalar, 1)

    raise TransformError("unknown constant type", expr)

def transform_multiply_elementwise(expr):
    if len(expr.arg) != 2:
        raise TransformError("wrong number of args", expr)

    if expr.arg[0].dcp_props.constant:
        c_expr = expr.arg[0]
        x_expr = expr.arg[1]
    elif expr.arg[1].dcp_props.constant:
        c_expr = expr.arg[1]
        x_expr = expr.arg[0]
    else:
        raise TransformError("multiply non constants", expr)

    return expression.linear_map(
        multiply_elementwise_constant(c_expr),
        transform_expr(x_expr))

def transform_negate(expr):
    return expression.linear_map(
        linear_map.negate(dim(expr)),
        transform_expr(only_arg(expr)))

def transform_sum(expr):
    x = only_arg(expr)
    m, n = dims(x)

    if not expr.has_axis:
        return expression.linear_map(
            linear_map.sum(m, n),
            transform_expr(x))

    if expr.axis == 0:
        return expression.linear_map(
            linear_map.sum_left(m, n),
            transform_expr(x))

    if expr.axis == 1:
        return expression.linear_map(
            linear_map.sum_right(m, n),
            transform_expr(x))

    raise TransformError("unknown axis attribute", expr)

def transform_hstack(expr):
    m = dim(expr, 0)
    n = dim(expr, 1)
    offset = 0
    add_args = []
    for arg in expr.arg:
        ni = dim(arg, 1)
        add_args.append(
            expression.linear_map(
                linear_map.right_matrix_product(
                    linear_map.index(slice(offset, offset+ni), n), m),
                transform_expr(arg)))
        offset += ni
    return expression.add(*add_args)

def transform_vstack(expr):
    m = dim(expr, 0)
    n = dim(expr, 1)
    offset = 0
    add_args = []
    for arg in expr.arg:
        mi = dim(arg, 0)

        add_args.append(
            expression.linear_map(
                linear_map.left_matrix_product(
                    linear_map.transpose(
                        linear_map.index(slice(offset, offset+mi), m)),
                    n),
                transform_expr(arg)))
        offset += mi
    return expression.add(*add_args)

def transform_reshape(expr):
    # drop reshape nodes as everything is a vector
    return transform_expr(only_arg(expr))

def transform_linear_map(expr):
    return expr

def transform_diag_mat(expr):
    return expression.linear_map(
        linear_map.diag_mat(dim(expr)),
        transform_expr(only_arg(expr)))

def transform_diag_vec(expr):
    return expression.linear_map(
        linear_map.diag_vec(dim(expr, 0)),
        transform_expr(only_arg(expr)))

def transform_upper_tri(expr):
    return expression.linear_map(
        linear_map.upper_tri(dim(expr, 0)),
        transform_expr(only_arg(expr)))

def transform_trace(expr):
    return expression.linear_map(
        linear_map.trace(dim(only_arg(expr), 0)),
        transform_expr(only_arg(expr)))

def transform_power(expr):
    p = expr.p
    if p == 1:
        return transform_expr(only_arg(expr))
    if p == 0:
        return expression.scalar_constant(1)

    raise TransformError("Unexpected power exponent", expr)

def transform_transpose(expr):
    x = only_arg(expr)
    return expression.linear_map(
        linear_map.transpose_matrix(*dims(x)),
        transform_expr(x))

def transform_linear_expr(expr):
    log_debug_expr("transform_linear_expr", expr)
    f_name = "transform_" + Expression.Type.Name(expr.expression_type).lower()
    return globals()[f_name](expr)

def transform_expr(expr):
    if expr.func_curvature.curvature_type in (
            Curvature.AFFINE,
            Curvature.CONSTANT):
        return transform_linear_expr(expr)
    else:
        transformed_expr = expression.Expression()
        transformed_expr.proto.CopyFrom(expr.proto)
        for arg in expr.arg:
            transformed_expr.arg.append(transform_expr(arg))
        return transformed_expr

def transform_problem(problem):
    return Problem(
        objective=transform_expr(problem.objective),
        constraint=[transform_expr(e) for e in problem.constraint])
