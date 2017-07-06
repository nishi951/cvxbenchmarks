
from collections import namedtuple

import numpy as np
import cvxpy as cp
from numpy.random import randn, rand

from epopt.proto.epsilon.expression_pb2 import ProxFunction
from epopt.prox import eval_prox_impl

RANDOM_PROX_TRIALS = 10

import logging
logging.basicConfig(level=logging.DEBUG)

# Common variable
n = 10
x = cp.Variable(n)
z = cp.Variable(n)
p = cp.Variable(3)
q = cp.Variable(3)
X = cp.Variable(3,3)
t = cp.Variable(1)
p1 = cp.Variable(1)
q1 = cp.Variable(1)

t_vec = cp.Variable(3,1)
t_rvec = cp.Variable(1,3)

ProxTest = namedtuple(
    "ProxTest", ["prox_function_type", "objective", "constraint", "epigraph"])

def prox(prox_function_type, objective, constraint=None):
    return ProxTest(prox_function_type, objective, constraint, False)

def epigraph(prox_function_type, objective, constraint):
    return ProxTest(prox_function_type, objective, constraint, True)

def f_quad_form():
    m = 4
    x = cp.Variable(m)
    A = np.random.randn(m, m)
    P = np.identity(m)*0.001
    return cp.quad_form(x, P)

def f_quantile():
    alpha = rand()
    return cp.sum_entries(cp.max_elemwise(alpha*x,(alpha-1)*x))

def f_quantile_elemwise():
    m = 4
    k = 2
    alphas = rand(k)
    A = np.tile(alphas, (m, 1))
    X = cp.Variable(m, k)
    return cp.sum_entries(cp.max_elemwise(
        cp.mul_elemwise( -A, X),
        cp.mul_elemwise(1-A, X)))

def f_dead_zone():
    eps = np.abs(randn())
    return cp.sum_entries(cp.max_elemwise(cp.abs(x)-eps, 0))

def f_hinge():
    return cp.sum_entries(cp.max_elemwise(x,0))

def f_hinge_axis0():
    return cp.sum_entries(cp.max_elemwise(X,0), axis=0)

def f_hinge_axis1():
    return cp.sum_entries(cp.max_elemwise(X,0), axis=1)

def f_least_squares(m):
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    return cp.sum_squares(A*x - b)

def f_least_squares_matrix():
    m = 20
    k = 3
    A = np.random.randn(m, n)
    B = np.random.randn(m, k)
    X = cp.Variable(n, k)
    return cp.sum_squares(A*X  - B)

def f_norm1_weighted():
    w = np.random.randn(n)
    w[0] = 0
    return cp.norm1(cp.mul_elemwise(w, x))

def C_linear_equality():
    m = 5
    A = np.random.randn(m, n)
    b = A.dot(np.random.randn(n))
    return [A*x == b]

def C_linear_equality_matrix_lhs():
    m = 5
    k = 3
    A = np.random.randn(m, n)
    X = cp.Variable(n, k)
    B = A.dot(np.random.randn(n, k))
    return [A*X == B]

def C_linear_equality_matrix_rhs():
    m = 3
    k = 5
    A = np.random.randn(k, m)
    X = cp.Variable(n, k)
    B = np.random.randn(n, k).dot(A)
    return [X*A == B]

def C_linear_equality_graph(m):
    A = np.random.randn(m, n)
    y = cp.Variable(m)
    return [y == A*x]

def C_linear_equality_graph_lhs(m, n):
    k = 3
    A = np.random.randn(m, n)
    B = A.dot(np.random.randn(n,k))
    X = cp.Variable(n, k)
    Y = cp.Variable(m, k)
    return [Y == A*X + B]

def C_linear_equality_graph_rhs(m, n):
    k = 3
    A = np.random.randn(m, n)
    B = np.random.randn(k, m).dot(A)
    X = cp.Variable(k, m)
    Y = cp.Variable(k, n)
    return [Y == X*A + B]

def C_linear_equality_multivariate():
    m = 5
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    alpha = np.random.randn()
    y = cp.Variable(m)
    z = cp.Variable(m)
    return [z - (y - alpha*(A*x - b)) == 0]

def C_linear_equality_multivariate2():
    m = 5
    A = np.random.randn(m, n)
    y = cp.Variable(m)
    z = cp.Variable(m)
    return [z - (y - (1 - A*x)) == 0]

def C_non_negative_scaled():
    alpha = np.random.randn()
    return [alpha*x >= 0]

def C_non_negative_scaled_elemwise():
    alpha = np.random.randn(n)
    return [cp.mul_elemwise(alpha, x) >= 0]

def C_soc_scaled():
    return [cp.norm2(randn()*x) <= randn()*t]

def C_soc_translated():
    return [cp.norm2(x + randn()) <= t + randn()]

def C_soc_scaled_translated():
    return [cp.norm2(randn()*x + randn()) <= randn()*t + randn()]

# Proximal operators
PROX_TESTS = [
    #prox("MATRIX_FRAC", lambda: cp.matrix_frac(p, X)),
    #prox("SIGMA_MAX", lambda: cp.sigma_max(X)),
    prox("AFFINE", lambda: randn(n).T*x),
    prox("CONSTANT", lambda: 0),
    prox("LAMBDA_MAX", lambda: cp.lambda_max(X)),
    prox("LOG_SUM_EXP", lambda: cp.log_sum_exp(x)),
    prox("MAX", lambda: cp.max_entries(x)),
    prox("NEG_LOG_DET", lambda: -cp.log_det(X)),
    prox("NON_NEGATIVE", None, C_non_negative_scaled),
    prox("NON_NEGATIVE", None, C_non_negative_scaled_elemwise),
    prox("NON_NEGATIVE", None, lambda: [x >= 0]),
    prox("NORM_1", f_norm1_weighted),
    prox("NORM_1", lambda: cp.norm1(x)),
    prox("NORM_2", lambda: cp.norm(X, "fro")),
    prox("NORM_2", lambda: cp.norm2(x)),
    prox("NORM_NUCLEAR", lambda: cp.norm(X, "nuc")),
    #prox("QUAD_OVER_LIN", lambda: cp.quad_over_lin(p, q1)),
    prox("SECOND_ORDER_CONE", None, C_soc_scaled),
    prox("SECOND_ORDER_CONE", None, C_soc_scaled_translated),
    prox("SECOND_ORDER_CONE", None, C_soc_translated),
    prox("SECOND_ORDER_CONE", None, lambda: [cp.norm(X, "fro") <= t]),
    prox("SECOND_ORDER_CONE", None, lambda: [cp.norm2(x) <= t]),
    prox("SEMIDEFINITE", None, lambda: [X >> 0]),
    prox("SUM_DEADZONE", f_dead_zone),
    prox("SUM_EXP", lambda: cp.sum_entries(cp.exp(x))),
    prox("SUM_HINGE", f_hinge),
    prox("SUM_HINGE", lambda: cp.sum_entries(cp.max_elemwise(1-x, 0))),
    prox("SUM_HINGE", lambda: cp.sum_entries(cp.max_elemwise(1-x, 0))),
    prox("SUM_INV_POS", lambda: cp.sum_entries(cp.inv_pos(x))),
    prox("SUM_KL_DIV", lambda: cp.sum_entries(cp.kl_div(p1,q1))),
    prox("SUM_LARGEST", lambda: cp.sum_largest(x, 4)),
    prox("SUM_LOGISTIC", lambda: cp.sum_entries(cp.logistic(x))),
    prox("SUM_NEG_ENTR", lambda: cp.sum_entries(-cp.entr(x))),
    prox("SUM_NEG_LOG", lambda: cp.sum_entries(-cp.log(x))),
    prox("SUM_QUANTILE", f_quantile),
    prox("SUM_QUANTILE", f_quantile_elemwise),
    prox("SUM_SQUARE", f_least_squares_matrix),
    prox("SUM_SQUARE", lambda: f_least_squares(20)),
    prox("SUM_SQUARE", lambda: f_least_squares(5)),
    prox("SUM_SQUARE", f_quad_form),
    prox("TOTAL_VARIATION_1D", lambda: cp.tv(x)),
    prox("ZERO", None, C_linear_equality),
    prox("ZERO", None, C_linear_equality_matrix_lhs),
    prox("ZERO", None, C_linear_equality_matrix_rhs),
    prox("ZERO", None, C_linear_equality_multivariate),
    prox("ZERO", None, C_linear_equality_multivariate2),
    prox("ZERO", None, lambda: C_linear_equality_graph(20)),
    prox("ZERO", None, lambda: C_linear_equality_graph(5)),
    prox("ZERO", None, lambda: C_linear_equality_graph_lhs(10, 5)),
    prox("ZERO", None, lambda: C_linear_equality_graph_lhs(5, 10)),
    prox("ZERO", None, lambda: C_linear_equality_graph_rhs(10, 5)),
    prox("ZERO", None, lambda: C_linear_equality_graph_rhs(5, 10)),
]

# Epigraph operators
PROX_TESTS += [
    epigraph("NEG_LOG_DET", None, lambda: [-cp.log_det(X) <= t]),
    epigraph("EXP", None, lambda: [cp.exp(x) <= z]),
    epigraph("LOG_SUM_EXP", None, lambda: [cp.log_sum_exp(x) <= t]),
    epigraph("LOG_SUM_EXP", None, lambda: [cp.log_sum_exp(X, axis=0) <= t_rvec]),
    epigraph("LOG_SUM_EXP", None, lambda: [cp.log_sum_exp(X, axis=1) <= t_vec]),
    epigraph("LAMBDA_MAX", None, lambda: [cp.lambda_max(X) <= t]),
    epigraph("MAX", None, lambda: [cp.max_entries(x) <= t]),
    epigraph("NORM_1", None, lambda: [cp.norm1(x) <= t]),
    epigraph("NORM_NUCLEAR", None, lambda: [cp.norm(X, "nuc") <= t]),
    epigraph("SUM_DEADZONE", None, lambda: [f_dead_zone() <= t]),
    epigraph("SUM_EXP", None, lambda: [cp.sum_entries(cp.exp(x)) <= t]),
    epigraph("SUM_HINGE", None, lambda: [f_hinge() <= t]),
    epigraph("SUM_HINGE", None, lambda: [f_hinge_axis0() <= t_rvec]),
    epigraph("SUM_HINGE", None, lambda: [f_hinge_axis1() <= t_vec]),
    epigraph("SUM_INV_POS", None, lambda: [cp.sum_entries(cp.inv_pos(x)) <= t]),
    epigraph("SUM_KL_DIV", None, lambda: [cp.sum_entries(cp.kl_div(p1,q1)) <= t]),
    epigraph("SUM_LARGEST", None, lambda: [cp.sum_largest(x, 4) <= t]),
    epigraph("SUM_LOGISTIC", None, lambda: [cp.sum_entries(cp.logistic(x)) <= t]),
    epigraph("SUM_NEG_ENTR", None, lambda: [cp.sum_entries(-cp.entr(x)) <= t]),
    epigraph("SUM_NEG_LOG", None, lambda: [cp.sum_entries(-cp.log(x)) <= t]),
    epigraph("SUM_QUANTILE", None, lambda: [f_quantile() <= t]),
    #epigraph("SUM_SQUARE", None, lambda: [f_quad_form() <= t]),
    epigraph("SUM_SQUARE", None, lambda: [cp.sum_squares(x) <= t]),
]

def run_prox(prox_function_type, prob, v_map, lam=1, epigraph=False):
    eval_prox_impl(prob, v_map, lam, prox_function_type, epigraph)
    actual = {x: x.value for x in prob.variables()}

    # Compare to solution with cvxpy
    prob.objective.args[0] *= lam
    prob.objective.args[0] += sum(
        0.5*cp.sum_squares(x - v_map[x]) for x, v in v_map.items())
    try:
        prob.solve()
    except cp.SolverError as e:
        # If CVXPY fails with default, try again with SCS
        prob.solve(solver=cp.SCS)

    try:
        for x in prob.variables():
            np.testing.assert_allclose(x.value, actual[x], rtol=1e-2, atol=1e-2)
    except AssertionError as e:
        # print objective value and constraints
        print()
        print('cvx:')
        print([x.value for x in prob.variables()])
        print('actual:')
        print(list(actual.values()))
        print('vmap:')
        print(list(v_map.values()))
        print('cvx obj:', prob.objective.value)
        for c in prob.constraints:
            print(c, c.value, [x.value for x in c.args])

        for x,v in list(actual.items()):
            x.value = v
            print('our obj:', prob.objective.value)
        for c in prob.constraints:
            print(c, c.value, [x.value for x in c.args])
        print()

        raise e

def run_random_prox(prox_test, trial):
    np.random.seed(trial)
    v = np.random.randn(n)
    lam = np.abs(np.random.randn())

    f = 0 if not prox_test.objective else prox_test.objective()
    C = [] if not prox_test.constraint else prox_test.constraint()

    # Form problem and solve with proximal operator implementation
    prob = cp.Problem(cp.Minimize(f), C)
    v_map = {x: np.random.randn(*x.size) for x in prob.variables()}

    t = ProxFunction.Type.Value(prox_test.prox_function_type)
    run_prox(t, prob, v_map, lam, prox_test.epigraph)

def test_random_prox():
    for prox in PROX_TESTS:
        for trial in range(RANDOM_PROX_TRIALS):
            yield run_random_prox, prox, trial

def test_second_order_cone():
    v_maps = [
        {x: np.zeros(10), t: np.array([0])},
        {x: np.arange(10), t: np.array([100])},
        {x: np.arange(10), t: np.array([10])},
        {x: np.arange(10), t: np.array([-100])},
        {x: np.arange(10), t: np.array([-10])}]

    for v_map in v_maps:
        prob = cp.Problem(cp.Minimize(0), [cp.norm(x) <= t])
        yield run_prox, ProxFunction.SECOND_ORDER_CONE, prob, v_map
