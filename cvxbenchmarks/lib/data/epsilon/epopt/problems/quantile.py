import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(m, n, k, p=1, sigma=0.1):
    # Generate data
    x = np.random.rand(m)*2*np.pi*p
    y = np.sin(x) + sigma*np.sin(x)*np.random.randn(m)
    alphas = np.linspace(1./(k+1), 1-1./(k+1), k)

    # RBF features
    mu_rbf = np.array([np.linspace(-1, 2*np.pi*p+1, n)])
    mu_sig = (2*np.pi*p+2)/n
    X = np.exp(-(mu_rbf.T - x).T**2/(2*mu_sig**2))

    Theta = cp.Variable(n,k)
    f = ep.quantile_loss(alphas, Theta, X, y)
    C = [X*(Theta[:,:-1] - Theta[:,1:]) >= 0]
    return cp.Problem(cp.Minimize(f), C)
