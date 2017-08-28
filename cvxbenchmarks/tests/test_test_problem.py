# test_test_problem.py

import pytest
from mock import patch, call, mock_open, MagicMock, Mock

import cvxbenchmarks.framework as t
import cvxbenchmarks.settings as s
import cvxpy as cp
import numpy as np

####################
# Problem Fixtures #
####################
# A bunch of cvxpy problems of different types.

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


def test_testproblem_init(lp_prob, socp_prob, sdp_prob):
    p1 = t.TestProblem("lp_prob", lp_prob)
    assert p1.tags == set([s.LP])
    p2 = t.TestProblem("socp_prob", socp_prob)
    assert p2.tags == set([s.LP, s.SOCP])
    p3 = t.TestProblem("sdp_prob", sdp_prob)
    assert p3.tags == set([s.LP, s.SOCP, s.SDP])

def test_testproblem_get_all_from_file():
    pass





