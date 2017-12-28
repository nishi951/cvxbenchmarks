import pytest
from mock import patch, call, mock_open, MagicMock, Mock

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
# This unit test requries the file test_config.yml, with the following contents:
#
# solver: SOLVER
# verbose: true
# eps: 1e-4
#

def test_read_and_configure():
    config1 = CVXConfig.from_file(os.path.join(str(pytest.config.rootdir), 
                                               "cvxbenchmarks", "cvx", "tests", "test_config.yml"))
    assert config1.configure() == {"solver": "SOLVER", "eps": 1e-4, "verbose": True}
    assert config1.configID == "CONFIGID"





