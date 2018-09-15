import pytest
from mock import patch, call, mock_open, MagicMock, Mock

import cvxbenchmarks.cvx.cvxconfig
from cvxbenchmarks.cvx.cvxconfig import CVXConfig
import cvxbenchmarks.settings as s
import cvxpy as cp
import numpy as np

import sys, os.path

def test_init():
    config1 = CVXConfig("config1")
    assert config1.configID == "config1"
    assert config1.solver_opts == {}
    config2 = CVXConfig("config2", {"test": 1})
    assert config2.configID == "config2"
    assert config2.solver_opts == {"test": 1}

#############
# IMPORTANT #
#############
# This unit test requries the file tests/test_config.yml.


def test_read_YAML_and_configure():
    configs = CVXConfig.read(os.path.join(str(pytest.config.rootdir), 
                                               "cvxbenchmarks", "cvx", "tests", "configs", "test_config.yml"))
    assert configs[0].configure() == {"solver": "solver1", "eps": 1e-4, "verbose": True}
    assert configs[0].configID == "config1"
    assert configs[1].configure() == {"solver": "solver2", "eps": 1e-4, "verbose": True}
    assert configs[1].configID == "config2"





