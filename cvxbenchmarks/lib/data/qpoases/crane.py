# Generate QPs from crane
# Mark Nishimura 2017

import numpy as np
import cvxpy as cp
import scipy as sp

import os.path

root = "crane"

dims = np.loadtxt(os.path.join(root, "dims.oqp"))
nQP = int(dims[0]) # Number of QPs
nV = int(dims[1])  # Number of variables
nC = int(dims[2])  # Number of constraints
nEC = int(dims[3]) # Number of equality constraints

H = np.loadtxt(os.path.join(root, "H.oqp")) # Hessian; Constant for all QPs
A = np.loadtxt(os.path.join(root, "A.oqp")) # Constraint matrix; Constant for all QPs
g = np.loadtxt(os.path.join(root, "g.oqp")) # gradient term; differs for each QP
lb = np.loadtxt(os.path.join(root, "lb.oqp")) # lower bounds on x
ub = np.loadtxt(os.path.join(root, "ub.oqp")) # upper bounds on x
lbA = np.loadtxt(os.path.join(root, "lbA.oqp")) # lower bounds on Ax
ubA = np.loadtxt(os.path.join(root, "ubA.oqp")) # upper bounds on Ax

obj_opt = np.loadtxt(os.path.join(root, "obj_opt.oqp")) # optimal values

# Form problems:
probs = []
for i in range(nQP):
    x = cp.Variable(nV)
    obj = cp.Minimize(0.5*cp.quad_form(x, H) + g[i]*x)
    constr = [lb[i] <= x, x <= ub[i], lbA[i] <= A*x, A*x <= ubA[i]]
    probs += [cp.Problem(obj, constr)]

opt_vals = list(obj_opt)



if __name__ == "__main__":
    for i, prob in enumerate(probs):
        prob.solve(solver="MOSEK")
        opt_val = opt_vals[i]
        print("Problem: " + str(i))
        print("\tstatus: " + str(prob.status))
        print("\toptimal value: " + str(prob.value))
        print("\ttrue optimal value: " + str(opt_val))
