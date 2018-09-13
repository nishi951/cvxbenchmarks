import pytest
#from mock import patch, call, mock_open, MagicMock, Mock
import numpy as np
import cvxpy as cp
import pandas as pd
import os

from cvxbenchmarks.cvx.cvxproblem import CVXProblem
from cvxbenchmarks.cvx.cvxconfig import CVXConfig
from cvxbenchmarks.cvx.cvxcore import CVXInstance, CVXResults
from cvxbenchmarks.cvx.cvxframework import CVXFramework

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

#############
# Important #
#############
# This requires there to be a folder called "tests/problems"

@pytest.fixture
def problem_path():
    return str(os.path.join(str(pytest.config.rootdir), 
           "cvxbenchmarks", "cvx", "tests", "problems"))

#############
# Important #
#############
# This requires there to be a folder called "tests/configs"
@pytest.fixture
def config_path():
    return str(os.path.join(str(pytest.config.rootdir), 
               "cvxbenchmarks", "cvx", "tests", "configs"))

def test_init_framework():
    f1 = CVXFramework()
    assert f1.problems == []
    assert f1.configs == []
    assert f1.instances == []
    assert f1.results == []
    assert f1.cache == None

    f2 = CVXFramework([("problem1", "problems")],
                      [("config1", "configs")],
                      ["ticket"],
                      ["instance"],
                      ["result"],
                      "cache")
    assert f2.problems == [("problem1", "problems")]
    assert f2.configs == [("config1", "configs")]
    assert f2.tickets == ["ticket"]
    assert f2.instances == ["instance"]
    assert f2.results == ["result"]
    assert f2.cache == "cache"


def test_load_problem():
    f1 = CVXFramework()
    f1.load_problem("problem", "problemDir")
    assert f1.problems == [("problem", "problemDir")]

def test_load_config():
    f1 = CVXFramework()
    f1.load_config("config1", "configDir")
    assert f1.configs == [("config1", "configDir")]


def test_load_all_problems(problem_path):
    f1 = CVXFramework()
    f1.load_all_problems(problem_path)
    assert f1.problems == [("test_problem", problem_path)]


def test_load_all_configs(config_path):
    f1 = CVXFramework()
    f1.load_all_configs(config_path)
    assert f1.configs == [("test_config", config_path)]

def test_generate_tickets(problem_path, config_path):
    f1 = CVXFramework()
    f1.load_all_problems(problem_path)
    f1.load_all_configs(config_path)
    f1.generate_tickets()
    print(f1.tickets)

