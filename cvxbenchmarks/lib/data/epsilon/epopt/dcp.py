"""Compute DCP attributes of expressions.

This translates some of the functionality from cvxpy.atoms.atom.py and
cvxpy.expresison.expression.py to work on Epsilon's expression trees with the
help of cvxpy.utilities.
"""

import cvxpy.utilities

from epopt.proto.epsilon import expression_pb2

class DCPProperties(object):
    def __init__(self, dcp_attr):
        self.dcp_attr = dcp_attr
        self.curvature = expression_pb2.Curvature(
            curvature_type=expression_pb2.Curvature.Type.Value(
                dcp_attr.curvature.curvature_str))

    @property
    def affine(self):
        return (self.dcp_attr.curvature == cvxpy.utilities.Curvature.AFFINE or
                self.dcp_attr.curvature == cvxpy.utilities.Curvature.CONSTANT)

    @property
    def constant(self):
        return self.dcp_attr.curvature == cvxpy.utilities.Curvature.CONSTANT

from epopt import tree_format

def compute_dcp_properties(expr):
    # TODO(mwytock): Handle all unary/binary operators in this fashion.
    if expr.expression_type == expression_pb2.Expression.NEGATE:
        dcp_attr = -expr.arg[0].dcp_props.dcp_attr
    else:
        dcp_attr = cvxpy.utilities.DCPAttr(
            compute_sign(expr), compute_curvature(expr), compute_shape(expr))

    props = DCPProperties(dcp_attr)
    # print "compute_dcp_properties"
    # print tree_format.format_expr(expr)
    # print props.dcp_attr.curvature
    return props

def compute_sign(expr):
    return cvxpy.utilities.Sign(
        expression_pb2.Sign.Type.Name(expr.sign.sign_type))

def compute_shape(expr):
    return cvxpy.utilities.Shape(expr.size.dim[0], expr.size.dim[1])

def compute_curvature(expr):
    """Compute curvature based on DCP rules, from cvxpy.atoms.atom"""
    func_curvature = cvxpy.utilities.Curvature(
        expression_pb2.Curvature.Type.Name(expr.func_curvature.curvature_type))
    if not expr.arg:
        return func_curvature

    if expr.arg_monotonicity:
        ms = [expression_pb2.Monotonicity.Type.Name(m.monotonicity_type)
              for m in expr.arg_monotonicity]
        assert len(ms) == len(expr.arg)
    else:
        # Default
        ms = [cvxpy.utilities.monotonicity.NONMONOTONIC]*len(expr.arg)

    return reduce(
        lambda a, b: a+b,
        (cvxpy.utilities.monotonicity.dcp_curvature(
            monotonicity,
            func_curvature,
            arg.dcp_props.dcp_attr.sign,
            arg.dcp_props.dcp_attr.curvature)
         for arg, monotonicity in zip(expr.arg, ms)))
