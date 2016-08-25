"""Split inefficient expressions."""

from epopt import affine
from epopt import expression
from epopt import tree_format
from epopt.compiler import validate
from epopt.compiler.transforms import linear
from epopt.compiler.transforms.transform_util import *
from epopt.proto.epsilon.expression_pb2 import Expression

def transform_linear_map(expr, constrs):
    if expr.arg[0].expression_type == Expression.LINEAR_MAP:
        # TODO(mwytock): need more complete logic here, i.e. need to do type
        # inference across the entire chain using something like affine_props
        A = affine.LinearMapType(expr.linear_map)
        B = affine.LinearMapType(expr.arg[0].linear_map)

        if ((A.kronecker_product or B.kronecker_product) and
            not (A*B).kronecker_product):
            t, epi_constrs = epi_transform(expr.arg[0], "split_linear_map")
            constrs += [linear.transform_expr(x) for x in epi_constrs]
            return expression.from_proto(expr.proto, [t], expr.data)

    return expr

def transform_expr(expr, constrs):
    expr = expression.from_proto(
        expr.proto,
        [transform_expr(arg, constrs) for arg in expr.arg],
        expr.data)

    f_name = "transform_" + Expression.Type.Name(expr.expression_type).lower()
    if f_name in globals():
        return globals()[f_name](expr, constrs)

    return expr

def transform_problem(problem):
    validate.check_sum_of_prox(problem)
    constrs = []
    obj = transform_expr(problem.objective, constrs)
    return expression.Problem(objective=obj, constraint=constrs)
