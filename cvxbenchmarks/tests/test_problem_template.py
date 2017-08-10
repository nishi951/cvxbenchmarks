import pytest
from mock import patch, call, mock_open

import pandas as pd
from jinja2 import Environment, FileSystemLoader
import io

from cvxbenchmarks.problem_generator import ProblemTemplate


@pytest.fixture
def template(request):
    TEMPLATE = \
    """
test
{{ problemID }}
{{ a }}
{{ b }}
    """
    environment = Environment()
    return environment.from_string(TEMPLATE)



@pytest.fixture
def paramsDf(request):
    PARAMS = u"""\
id,a,b
p1,1,2
p2,3,4
#p3,5,6
    """
    paramsDf = pd.read_csv(io.StringIO(PARAMS), skipinitialspace=True, comment='#')
    return paramsDf

@pytest.fixture
def paramsList(request):
    PARAMS_DICT = [
    {
        "problemID": "test_problem_p1",
        "a": 1,
        "b": 2,
        "id": "p1"
    },
    {
        "problemID": "test_problem_p2",
        "a": 3,
        "b": 4,
        "id": "p2"
    }
    ]
    return PARAMS_DICT

@pytest.fixture
def fullTemplate1(request):
    RENDERED = \
    u"""
test
test_problem_p1
1
2
    """
    return RENDERED

@pytest.fixture
def fullTemplate2(request):
    RENDERED = \
    u"""
test
test_problem_p2
3
4
    """
    return RENDERED

@pytest.fixture
def emptyTemplate(request):
    EMPTY = \
    u"""
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

@patch('cvxbenchmarks.problem_generator.Environment.get_template')
@patch('cvxbenchmarks.problem_generator.pd.read_csv')
def test_from_file(mock_read_csv, mock_get_template, paramsDf, template, paramsList):
    # Set up mock return values with fixtures.
    mock_get_template.return_value = template
    mock_read_csv.return_value = paramsDf

    templ = ProblemTemplate.from_file("templateName", "params", name="test_problem")
    assert templ.name == "test_problem"
    assert templ.template == template
    assert templ.params == paramsList
    return

@patch('cvxbenchmarks.problem_generator.Environment.get_template')
def test_read_template(mock_get_template, template):
    mock_get_template.return_value = template

    templ = ProblemTemplate(template = "template")
    templ.read_template("someDir/templ")
    assert templ.template == template

    mock_get_template.side_effect = Exception()
    templ = ProblemTemplate(template="template")
    templ.read_template("someDir/othertempl")
    assert templ.template == None
    return

@patch('cvxbenchmarks.problem_generator.pd.read_csv')
def test_read_param_csv(mock_read_csv, paramsDf, paramsList):
    mock_read_csv.return_value = paramsDf
    templ = ProblemTemplate("templateFile", "paramFile", "test_problem")
    templ.read_param_csv("paramFile", overwrite=True)
    assert templ.params == paramsList

    templ.read_param_csv("paramFile", overwrite=False)
    assert templ.params == paramsList+paramsList

    # Test exception
    mock_read_csv.side_effect = Exception()
    templ.read_param_csv("paramFile", overwrite=True)
    assert templ.params == []
    return

@patch('cvxbenchmarks.problem_generator.pd.read_csv')
@patch('cvxbenchmarks.problem_generator.Environment.get_template')
def test_read(mock_get_template, mock_read_csv, template, paramsDf, paramsList):
    mock_get_template.return_value = template
    mock_read_csv.return_value = paramsDf

    templ = ProblemTemplate("template", "params", "test_problem")
    templ.read("templatefile", "paramfile", True)
    assert templ.params == paramsList
    assert templ.template == template

    templ.read("templatefile", "paramfile", False)
    assert templ.params == paramsList + paramsList
    assert templ.template == template
    

@patch('cvxbenchmarks.problem_generator.Environment.get_template')
@patch('cvxbenchmarks.problem_generator.pd.read_csv')
def test_write_to_dir(mock_read_csv, mock_get_template, template, paramsDf, fullTemplate1, fullTemplate2):
    mock_get_template.return_value = template
    mock_read_csv.return_value = paramsDf

    m = mock_open()
    # with patch('cvxbenchmarks.problem_generator.open', m, create=True):
    with patch('cvxbenchmarks.problem_generator.open', m, create=True):
        templ = ProblemTemplate.from_file("templateName", "params", name="test_problem")
        templ.write_to_dir("problemDir")
        expected_calls = [
            call("problemDir/test_problem_p1.py", "w"),
            call().__enter__(),
            call().write(fullTemplate1),
            call().__exit__(None, None, None),
            call("problemDir/test_problem_p2.py", "w"),
            call().__enter__(),
            call().write(fullTemplate2),
            call().__exit__(None, None, None)
        ]
        assert expected_calls == m.mock_calls

    


@patch('cvxbenchmarks.problem_generator.Environment.get_template')
@patch('cvxbenchmarks.problem_generator.pd.read_csv')
def test_str(mock_read_csv, mock_get_template, fullTemplate1, emptyTemplate, paramsDf, template):
    mock_get_template.return_value = template
    mock_read_csv.return_value = paramsDf
    newTemplate = ProblemTemplate.from_file("templateName", "params", name="test_problem")
    assert str(newTemplate) == fullTemplate1

    mock_read_csv.return_value = []
    newnewTemplate = ProblemTemplate.from_file("templateName2", "params2", name="test_problem")
    assert str(newnewTemplate) == emptyTemplate
    return