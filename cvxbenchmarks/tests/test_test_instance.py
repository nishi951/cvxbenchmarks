# test_test_instance

import pytest
from mock import patch, call, mock_open, MagicMock, Mock
import numpy as np
import pandas as pd

import cvxbenchmarks.framework as t
from collections import namedtuple


@pytest.fixture
def test_problem():
    problem = MagicMock()
    def solve_side_effect(**kwargs):
        return kwargs
    problem.solve.side_effect = solve_side_effect
    return t.TestProblem(problemID="testproblem",
                         problem=problem)

@pytest.fixture
def solver_configuration():
    config = {
        "solver": "solver",
        "verbose": True,
        "eps": 1e-5
    }
    return t.SolverConfiguration("solverconfig", config)



def test_test_instance_run(test_problem, solver_configuration):
    pass

def test_test_instance_eq():
    pass

def test_test_instance_hash():
    pass
