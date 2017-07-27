import pytest
import mock

import pandas as pd
from jinja2 import Environment, FileSystemLoader
import io

from cvxbenchmarks.problem_generator import ProblemTemplate


@pytest.fixture
def env(request):
    TEMPLATE = \
    """
    test
    {{ a }}
    {{ b }}
    """
    return Environment().from_string(TEMPLATE)



@pytest.fixture
def paramsDf(request):
    PARAMS = \
    """
    a, b
    1, 2
    3, 4
    """
    paramsDf = pd.read_csv(io.StringIO(PARAMS))
    return paramsDf



def test_init():
    ProblemTemplate()
    return

@mock.patch('cvxbenchmarks.problem_generator.Environment', autospec=True)
@mock.patch('cvxbenchmarks.problem_generator.pd.read_csv', autospec=True)
def test_from_file(mock_read_csv, mock_env, paramsDf, env):
    # Set up mock return values with fixtures.
    mock_env.get_template = mock.MagicMock(return_value = env)
    mock_read_csv.return_value = paramsDf

    newTemplate = ProblemTemplate.from_file("templateFile", "params")
    mock_env.get_template.assert_called_once_with("templateFile")
    mock_read_csv.assert_called_once_with("params", skipinitialspace=True)
    return

def test_read_template():
    pass

def test_read_param_csv():
    pass

def test_read():
    pass