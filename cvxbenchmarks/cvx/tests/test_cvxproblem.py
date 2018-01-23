import pytest
from mock import patch, call, mock_open, MagicMock, Mock

import cvxpy as cp
# import cvxbenchmarks.framework as t
import cvxbenchmarks.settings as s
import numpy as np

import sys

from cvxbenchmarks.cvx.cvxproblem import CVXProblem

@pytest.fixture
def lp_prob():
    # {LP}
    x = cp.Variable()
    return cp.Problem(cp.Minimize(x), [x >= 2])

@pytest.fixture
def socp_prob():
    # {LP, SOCP}
    x = cp.Variable(2)
    return cp.Problem(cp.Minimize(cp.norm(2*x - np.array([1,1]))))

@pytest.fixture
def sdp_prob():
    # {LP, SDP, SOCP}
    x = cp.Semidef(2)
    return cp.Problem(cp.Minimize(cp.norm(3*x, 'fro')))

@pytest.fixture
def exp_prob():
    # {LP, EXP}
    x = cp.Variable(2)
    return cp.Problem(cp.Minimize(cp.log_sum_exp(x)))

@pytest.fixture
def mip_prob():
    # {LP, MIP}
    x = cp.Int(1)
    return cp.Problem(cp.Minimize(x))

@pytest.fixture
def problems_list(lp_prob, socp_prob, sdp_prob, exp_prob, mip_prob):
    # a list of problem dictionaries:
    problems_list = [
    {
        "problemID": "lp_prob",
        "problem": lp_prob,
        "opt_val": None
    },
    {
        "problemID": "socp_prob",
        "problem": socp_prob,
        "opt_val": None
    },
    {
        "problemID": "sdp_prob",
        "problem": sdp_prob,
        "opt_val": None
    },
    {
        "problemID": "exp_prob",
        "problem": exp_prob,
        "opt_val": None
    },
    {
        "problemID": "mip_prob",
        "problem": mip_prob,
        "opt_val": None
    }
    ]
    return problems_list


def test_testproblem_init(problems_list):
    p1 = CVXProblem(**problems_list[0])
    assert p1.metadata["cone_types"] == set([s.LP])
    p2 = CVXProblem(**problems_list[1])
    assert p2.metadata["cone_types"] == set([s.LP, s.SOCP])
    p3 = CVXProblem(**problems_list[2])
    assert p3.metadata["cone_types"] == set([s.LP, s.SOCP, s.SDP])
    p4 = CVXProblem(**problems_list[3])
    assert p4.metadata["cone_types"] == set([s.LP, s.EXP])
    p5 = CVXProblem(**problems_list[4])
    assert p5.metadata["cone_types"] == set([s.LP, s.MIP])

# @patch("cvxbenchmarks.framework.__import__")
def test_read(problems_list):
    mock_import = MagicMock()
    mock_import.return_value = MagicMock(problems=problems_list)
    if sys.version_info[0] < 3: # Python 2.x
        with patch("__builtin__.__import__", mock_import):
            problems = CVXProblem.read("fileID", "problemDir")
            assert [problem.problemID for problem in problems] == ["lp_prob", "socp_prob", 
                                                            "sdp_prob", "exp_prob", "mip_prob"]
    else: # Python 3.x
        with patch("builtins.__import__", mock_import):
            problems = CVXProblem.read("fileID", "problemDir")
            assert [problem.problemID for problem in problems] == ["lp_prob", "socp_prob", 
                                                            "sdp_prob", "exp_prob", "mip_prob"]

    

