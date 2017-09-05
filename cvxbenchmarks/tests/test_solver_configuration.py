# test_solver_configuration.py

import pytest
from mock import patch, call, mock_open, MagicMock, Mock

import cvxbenchmarks.framework as t
import cvxbenchmarks.settings as s
import cvxpy as cp
import numpy as np

import sys

@pytest.fixture
def config_dict():
    config_dict = {
        "solver": "solver",
        "verbose": True,
        "arg": 1.0
    }
    return config_dict

@patch("cvxbenchmarks.framework.cvx.installed_solvers")
def test_solver_configuration_from_file(mock_installed_solvers, config_dict):
    mock_installed_solvers.return_value = ["solver"]
    mock_import = MagicMock()
    mock_import.return_value = MagicMock(name="configModule", config=config_dict)
    if sys.version_info[0] < 3: # Python 2.x
        with patch("__builtin__.__import__", mock_import):
            config = t.SolverConfiguration.from_file("configID", "configDir")
            assert config.configID == "configID"
            assert config.config == config_dict
    else: # Python 3.x
        with patch("builtins.__import__", mock_import):
            config = t.SolverConfiguration.from_file("configID", "configDir")
            assert config.configID == "configID"
            assert config.config == config_dict




