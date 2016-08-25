import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util
from epopt.functions import one_hot

def create(**kwargs):
    # m>k
    k = kwargs['k']  #class
    m = kwargs['m']  #instance
    n = kwargs['n']  #dim
    p = 5   #p-largest
    X = problem_util.normalized_data_matrix(m,n,1)
    Y = np.random.randint(0, k, m)

    Theta = cp.Variable(n,k)
    t = cp.Variable(1)
    texp = cp.Variable(m)
    f = t+cp.sum_largest(texp, p) + cp.sum_squares(Theta)
    C = []
    C.append(cp.log_sum_exp(X*Theta, axis=1) <= texp)
    Yi = one_hot(Y, k)
    C.append(-cp.sum_entries(cp.mul_elemwise(X.T.dot(Yi), Theta)) == t)

    t_eval = lambda: \
        -cp.sum_entries(cp.mul_elemwise(X.T.dot(one_hot(Y, k)), Theta)).value 
    f_eval = lambda: t_eval() \
        + cp.sum_largest(cp.log_sum_exp(X*Theta, axis=1), p).value \
        + cp.sum_squares(Theta).value
    
    return cp.Problem(cp.Minimize(f), C), f_eval
