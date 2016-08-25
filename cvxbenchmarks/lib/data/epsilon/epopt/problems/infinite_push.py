
from epopt.problems import problem_util
import epopt as ep
import cvxpy as cp

def create(m, n, d):
    Xp = problem_util.normalized_data_matrix(m, d, 1)
    Xn = problem_util.normalized_data_matrix(n, d, 1)
    lam = 1

    theta = cp.Variable(d)
    f = ep.infinite_push(theta, Xp, Xn) + lam*cp.sum_squares(theta)

    f_eval = lambda: f.value
    return cp.Problem(cp.Minimize(f)), f_eval
