
from nose.tools import assert_items_equal, assert_equal

from epopt import cvxpy_expr
from epopt.compiler import compiler
from epopt.compiler import validate
from epopt.problems import basis_pursuit
from epopt.problems import least_abs_dev
from epopt.problems import tv_1d
from epopt.problems import tv_denoise
from epopt.proto.epsilon.expression_pb2 import Expression, ProxFunction

Prox = ProxFunction

# temporary debugging
# import logging
# logging.basicConfig(level=logging.DEBUG)

def prox_ops(expr):
    retval = []
    for arg in expr.arg:
        retval += prox_ops(arg)
    if expr.expression_type == Expression.PROX_FUNCTION:
        retval.append(expr.prox_function.prox_function_type)
    return retval

def test_basis_pursuit():
    problem = compiler.compile_problem(cvxpy_expr.convert_problem(
        basis_pursuit.create(m=10, n=30)))
    assert_items_equal(
        prox_ops(problem.objective),
        [Prox.CONSTANT, Prox.NORM_1])
    assert_equal(2, len(problem.constraint))

def test_least_abs_deviations():
    problem = compiler.compile_problem(cvxpy_expr.convert_problem(
        least_abs_dev.create(m=10, n=5)))
    assert_items_equal(
        prox_ops(problem.objective),
        [Prox.CONSTANT, Prox.NORM_1])
    assert_equal(1, len(problem.constraint))

def test_tv_denoise():
    problem = compiler.compile_problem(cvxpy_expr.convert_problem(
        tv_denoise.create(n=10, lam=1)))
    assert_items_equal(
        prox_ops(problem.objective),
        3*[Prox.SUM_SQUARE] + [Prox.AFFINE] + [Prox.SECOND_ORDER_CONE])
    assert_equal(2, len(problem.constraint))

def test_tv_1d():
    problem = compiler.compile_problem(cvxpy_expr.convert_problem(
        tv_1d.create(n=10)))
    assert_items_equal(
        prox_ops(problem.objective),
        [Prox.TOTAL_VARIATION_1D] + [Prox.SUM_SQUARE])
    assert_equal(1, len(problem.constraint))
