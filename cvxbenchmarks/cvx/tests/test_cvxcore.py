import pytest
from mock import patch, call, mock_open, MagicMock, Mock
import numpy as np
import cvxpy as cp
import pandas as pd

from cvxbenchmarks.cvx.cvxproblem import CVXProblem
from cvxbenchmarks.cvx.cvxconfig import CVXConfig
from cvxbenchmarks.cvx.cvxcore import CVXInstance, CVXResults

#Debugging
from cvxpy.problems.problem_data.sym_data import SymData

@pytest.fixture
def empty_prob():
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(x))
    return CVXProblem(problemID="empty_prob",
                      problem=prob,
                      opt_val=float('-inf'))

@pytest.fixture
def lp_prob():
    # {LP}
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(x), [x >= 2])
    return CVXProblem(problemID="lp_prob",
                      problem=prob,
                      opt_val=2.0)

@pytest.fixture
def exp_prob():
    # {LP, EXP}
    x = cp.Variable(2)
    A = np.eye(2)
    prob = cp.Problem(cp.Minimize(cp.log_sum_exp(x)), [A*x >= 0])
    return CVXProblem(problemID="exp_prob",
                      problem=prob,
                      opt_val=float('-inf'))
@pytest.fixture
def mip_prob():
    # {LP, MIP}
    x = cp.Int(1)
    prob = cp.Problem(cp.Minimize(x),[x >= 0])
    return CVXProblem(problemID="mip_prob",
                      problem=prob,
                      opt_val=0.0)
    
@pytest.fixture
def scs_config():
    config = {
        "solver": "SCS",
        "verbose": True,
        "eps": 1e-3
    }
    return CVXConfig("SCS_config", config)

@pytest.fixture
def ecos_config():
    config = {
        "solver": "ECOS",
        "verbose": True,
        "eps": 1e-3
    }
    return CVXConfig("ECOS_config", config)



def test_instance_run_success(lp_prob, scs_config):
    instance = CVXInstance(lp_prob, scs_config)
    results = instance.run()
    assert results.problemID == "lp_prob"
    assert results.configID == "SCS_config"
    assert np.abs(results.opt_val - lp_prob.opt_val) <= scs_config.solver_opts["eps"]

def test_instance_run_failure(mip_prob, ecos_config):
    instance = CVXInstance(mip_prob, ecos_config)
    results = instance.run()
    assert results.problemID == "mip_prob"
    assert results.configID == "ECOS_config"
    assert results.opt_val is None # Problem was not solved.


def test_instance_check_compatibility(lp_prob, ecos_config, mip_prob, scs_config):
    should_work = CVXInstance(lp_prob, scs_config)
    should_fail = CVXInstance(mip_prob, ecos_config)
    assert should_work.check_compatibility()
    assert not should_fail.check_compatibility()


def test_instance_eq(lp_prob, ecos_config, scs_config):
    # TODO: Fix
    instance1 = CVXInstance(lp_prob, ecos_config)
    instance2 = CVXInstance(lp_prob, scs_config)
    instance3 = CVXInstance(lp_prob, scs_config)

    assert instance1 != instance2
    assert instance2 == instance3

def test_results_compute_residual_stats(empty_prob, lp_prob, scs_config, exp_prob):
    NoneOutput = CVXResults.compute_residual_stats(empty_prob.problem)
    assert NoneOutput == (None, None)

    CVXInstance(exp_prob, scs_config).run()
    CVXResults.compute_residual_stats(exp_prob.problem)

@pytest.fixture
def sample_results():
    results = pd.DataFrame(index = pd.MultiIndex.from_product([["prob1", "prob2", "prob3"],
                                                               ["mosek_config", "config1"]]),
                           columns = ["opt_val", "solve_time"])
    results.loc[("prob1", "mosek_config"),:] = [1.0, 0.1]
    results.loc[("prob1", "config1"),:] = [0.99, 1.0]
    results.loc[("prob2", "mosek_config"),:] = [5.0, 2.0]
    results.loc[("prob2", "config1"),:] = [6.0, np.NaN]
    results.loc[("prob3", "mosek_config")] = [np.NaN, np.NaN]
    results.loc[("prob3", "config1")] = [np.NaN, np.NaN]

    return results

def test_results_compute_mosek_error(sample_results):
    results = sample_results.copy()
    CVXResults.compute_mosek_error(results, "opt_val", "mosek_config")

def test_results_compute_performance(sample_results):
    results = sample_results.copy()
    CVXResults.compute_performance(results, "solve_time")
