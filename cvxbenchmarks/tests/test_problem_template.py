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
    environment = Environment()
    return environment.from_string(TEMPLATE)



@pytest.fixture
def paramsDf(request):
    PARAMS = """\
a,b
1,2
3,4
    """
    paramsDf = pd.read_csv(io.StringIO(PARAMS))
    return paramsDf

@pytest.fixture
def paramsList(request):
    PARAMS_DICT = [
    {
        "a": 1,
        "b": 2
    },
    {
        "a": 3,
        "b": 4
    }
    ]
    return PARAMS_DICT

@pytest.fixture
def fullTemplate(request):
    RENDERED = \
    """
test
1
2
    """
    return RENDERED

@pytest.fixture
def emptyTemplate(request):
    EMPTY = \
    """
test


    """
    return EMPTY


##################
# Test functions #
##################

def test_init():
    temp1 = ProblemTemplate(name="test1", template="template")
    assert temp1.name == "test1"
    assert temp1.template == "template"
    assert temp1.params == []

    temp2 = ProblemTemplate(name="test2", template="template", params="params")
    assert temp2.name == "test2"
    assert temp2.template == "template"
    assert temp2.params == "params"
    return

@mock.patch('cvxbenchmarks.problem_generator.Environment.get_template')
@mock.patch('cvxbenchmarks.problem_generator.pd.read_csv')
def test_from_file(mock_read_csv, mock_get_template, paramsDf, env, paramsList):
    # Set up mock return values with fixtures.
    mock_get_template.return_value = env
    mock_read_csv.return_value = paramsDf

    newTemplate = ProblemTemplate.from_file("templateName", "params", name="test_template")
    assert newTemplate.name == "templateName"
    assert newTemplate.template == env
    assert newTemplate.params == paramsList
    return

@mock.patch('cvxbenchmarks.problem_generator.Environment.get_template')
def test_read_template(mock_get_template):
    mock_get_template.side_effect = Exception()
    templ = ProblemTemplate(template="template")
    templ.read_template("someDir/othertempl")
    assert templ.template == None
    return

@mock.patch('cvxbenchmarks.problem_generator.pd.read_csv')
def test_read_param_csv(mock_read_csv, paramsDf, paramsList):
    mock_read_csv.return_value = paramsDf
    templ = ProblemTemplate("templateFile", "paramFile")
    templ.read_param_csv("paramFile", overwrite=True)
    assert templ.params == paramsList

    templ.read_param_csv("paramFile", overwrite=False)
    assert templ.params == paramsList+paramsList

    # Test exception
    mock_read_csv.side_effect = Exception()
    templ.read_param_csv("paramFile", overwrite=True)
    assert templ.params == []
    return

def test_read():
    pass

def test_write_to_dir():
    pass

@mock.patch('cvxbenchmarks.problem_generator.Environment.get_template')
@mock.patch('cvxbenchmarks.problem_generator.pd.read_csv')
def test_str(mock_read_csv, mock_get_template, fullTemplate, emptyTemplate, paramsDf, env):
    mock_get_template.return_value = env
    mock_read_csv.return_value = paramsDf
    newTemplate = ProblemTemplate.from_file("templateName", "params", name="test_template")
    assert str(newTemplate) == fullTemplate

    mock_read_csv.return_value = []
    newnewTemplate = ProblemTemplate.from_file("templateName2", "params2", name="test_template2")
    assert str(newnewTemplate) == emptyTemplate
    return