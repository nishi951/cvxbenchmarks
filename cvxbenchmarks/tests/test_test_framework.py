import pytest
from mock import patch, call, mock_open, MagicMock

import cvxbenchmarks.framework as t

##################
# Test Framework #
##################

@pytest.fixture
def default_parameters(request):
    return {
        "problemDir": "pdir1",
        "configDir": "cdir1"
    }

@pytest.fixture
def nondefault_parameters(request):
    return {
        "problemDir": "pdir2",
        "configDir": "cdir2",
        "problems": ["p1", "p2", "p3"],
        "configs": ["c1", "c2", "c3"],
        "cacheFile": "cache2.pkl",
        "parallel": True,
        "tags": ["SOCP", "SDP"],
        "instances": ["inst1"],
        "results": ["res1"]
    }

@pytest.fixture
def before_cache_pkl():
    return {hash(("prob1", "config1")): ("prob1", "config1"),
            hash(("prob1", "config2")): ("prob1", "config2")}

@pytest.fixture
def after_cache_pkl():
    results = [("prob1", "config1"),
           ("prob2", "config1"),
           ("prob1", "config2"),
           ("prob2", "config2")]
    all_results = {}
    for result in results:
        all_results[hash(result)] = result
    return all_results

def mock_testinstance():
    testinstance = MagicMock()
    def init_side_effect(problem, config):
        instance = MagicMock(name="testinstance_{}_{}".format(problem, config))
        instance.run.return_value = (problem, config)
        instance.__hash__.return_value = hash((problem, config))
        instance.testproblem.id = problem
        instance.config.id = config
        return instance # a mocked instance of class TestInstance
    testinstance.side_effect = init_side_effect
    return testinstance

def mock_testproblem():
    testproblem = MagicMock()
    def side_effect(problemID, problemDir):
        return [problemID]
    testproblem.get_all_from_file.side_effect = side_effect
    return testproblem

def mock_solverconfiguration():
    solverconfiguration = MagicMock()
    def side_effect(configID, configDir):
        return configID
    solverconfiguration.from_file.side_effect = side_effect
    return solverconfiguration


def test_testframework_init(default_parameters, nondefault_parameters):
    framework1 = t.TestFramework(**default_parameters)
    assert framework1.problemDir == "pdir1"
    assert framework1.configDir == "cdir1"
    assert framework1.problems == []
    assert framework1.configs == []
    assert framework1.cacheFile == "cache.pkl"
    assert framework1.parallel == False
    assert framework1.tags == []
    assert framework1.instances == []
    assert framework1.results == []

    framework2 = t.TestFramework(**nondefault_parameters)
    assert framework2.problemDir == "pdir2"
    assert framework2.configDir == "cdir2"
    assert framework2.problems == ["p1", "p2", "p3"]
    assert framework2.configs == ["c1", "c2", "c3"]
    assert framework2.cacheFile == "cache2.pkl"
    assert framework2.parallel == True
    assert framework2.tags == ["SOCP", "SDP"]
    assert framework2.instances == ["inst1"]
    assert framework2.results == ["res1"]

@patch("cvxbenchmarks.framework.TestProblem.get_all_from_file")
def test_testframework_load_problem_file(mock_get_all_from_file, 
                                         default_parameters, 
                                         nondefault_parameters):
    mock_get_all_from_file.return_value = ["problem1", "problem2"]
    framework1 = t.TestFramework(**default_parameters)
    framework1.load_problem_file("test_problem")
    assert framework1.problems == ["problem1", "problem2"]

    framework2 = t.TestFramework(**nondefault_parameters)
    framework2.load_problem_file("test_problem")
    assert framework2.problems == ["p1", "p2", "p3"] + ["problem1", "problem2"]

@patch("cvxbenchmarks.framework.os.walk")
@patch("cvxbenchmarks.framework.TestFramework.load_problem_file")
def test_testframework_preload_all_problems(mock_load_problem_file,
                                            mock_os_walk,
                                            default_parameters):
    mock_os_walk.return_value = [("cvxbenchmarks/problems", [], ["__init__.py", "problem1.py", "problem2.py"])]
    framework1 = t.TestFramework(**default_parameters)
    framework1.preload_all_problems()

    calls = mock_load_problem_file.mock_calls    
    assert call("problem1") in calls
    assert call("problem2") in calls
    assert not call("__init__") in calls

@patch("cvxbenchmarks.framework.SolverConfiguration.from_file")
def test_testframework_load_config(mock_from_file, 
                                    default_parameters):
    mock_from_file.side_effect = ["config1", "config2", None]
    framework1 = t.TestFramework(**default_parameters)
    framework1.load_config("configID1")
    assert framework1.configs == ["config1"]
    framework1.load_config("configID2")
    assert framework1.configs == ["config1", "config2"]
    framework1.load_config("configID_None")
    assert framework1.configs == ["config1", "config2"]

@patch("cvxbenchmarks.framework.os.walk")
@patch("cvxbenchmarks.framework.TestFramework.load_config")
def test_testframework_preload_all_configs(mock_load_config,
                                            mock_os_walk,
                                            default_parameters):
    mock_os_walk.return_value = [("cvxbenchmarks/lib/configs", [], ["__init__.py", "config1.py", "config2.py"])]
    framework1 = t.TestFramework(**default_parameters)
    framework1.preload_all_configs()

    calls = mock_load_config.mock_calls    
    assert call("config1") in calls
    assert call("config2") in calls
    assert not call("__init__") in calls

def test_testframework_generate_test_instances(default_parameters):
    problems = ["prob1", "prob2"]
    configs = ["config1", "config2"]
    framework1 = t.TestFramework(**default_parameters)
    framework1.problems = problems
    framework1.configs = configs
    framework1.generate_test_instances()
    assert framework1.instances == [t.TestInstance("prob1", "config1"),
                                    t.TestInstance("prob1", "config2"),
                                    t.TestInstance("prob2", "config1"),
                                    t.TestInstance("prob2", "config2")]

def test_testframework_clear_cache(default_parameters):
    import pickle as pkl
    m = mock_open()

    with patch("cvxbenchmarks.framework.open", m):
        framework1 = t.TestFramework(**default_parameters)
        framework1.clear_cache()
        expected_calls = [
            call("cache.pkl", "wb"),
            call().__enter__(),
            call().write(pkl.dumps({})),
            call().__exit__(None, None, None)
        ]
        assert expected_calls == m.mock_calls

@patch("cvxbenchmarks.framework.TestFramework.solve_all_parallel")
@patch("cvxbenchmarks.framework.TestFramework.solve_all")
def test_testframework_solve(mock_solve_all,
                             mock_solve_all_parallel,
                             default_parameters,
                             nondefault_parameters):
    framework1 = t.TestFramework(**default_parameters)
    framework1.solve(use_cache = True)
    mock_solve_all.assert_called_once_with(True)
    mock_solve_all_parallel.assert_not_called()

    framework2 = t.TestFramework(**nondefault_parameters)
    framework2.solve(use_cache = False)
    mock_solve_all.assert_called_once_with(True)
    mock_solve_all_parallel.assert_called_once_with(False)
    

@patch("cvxbenchmarks.framework.TestInstance", new_callable=mock_testinstance)
def test_testframework_solve_all_no_cache(mock_testinstance,
                                 default_parameters
                                 ):
    # Setup Mocks

    # Test without cache
    framework1 = t.TestFramework(**default_parameters)
    framework1.problems = ["prob1", "prob2"]
    framework1.configs = ["config1", "config2"]

    framework1.solve_all(use_cache = False)
    results = [("prob1", "config1"),
               ("prob2", "config1"),
               ("prob1", "config2"),
               ("prob2", "config2")]
    assert sorted(framework1.results) == sorted(results)


@patch("cvxbenchmarks.framework.pkl.dump")
@patch("cvxbenchmarks.framework.pkl.load")
@patch("cvxbenchmarks.framework.TestInstance", new_callable=mock_testinstance)
def test_testframework_solve_all_cache(mock_testinstance,
                                 mock_pickle_load,
                                 mock_pickle_dump,
                                 default_parameters,
                                 before_cache_pkl,
                                 after_cache_pkl
                                 ):


    # Save "old" cache for testing later
    from copy import deepcopy
    old_cache = deepcopy(before_cache_pkl)

    mock_pickle_load.return_value = before_cache_pkl
    m = mock_open()

    framework2 = t.TestFramework(**default_parameters)
    framework2.problems = ["prob1", "prob2"]
    framework2.configs = ["config1", "config2"]
    results = [("prob1", "config1"),
               ("prob2", "config1"),
               ("prob1", "config2"),
               ("prob2", "config2")]



    with patch("cvxbenchmarks.framework.open", m):
        framework2.solve_all(use_cache = True)
        assert sorted(framework2.results) == sorted(results)

        # Make sure results were not run if they were cached.
        for instance in framework2.instances:
            if hash(instance) in old_cache:
                instance.run.assert_not_called()


        # Make sure the final cache contains all results.
        assert call("cache.pkl", "wb") in m.mock_calls
        assert call(after_cache_pkl, m.return_value) in \
            mock_pickle_dump.mock_calls

# @patch("cvxbenchmarks.framework.SolverConfiguration")
# @patch("cvxbenchmarks.framework.TestProblem")
# @patch("cvxbenchmarks.framework.TestInstance")
# def test_testframework_solve_all_parallel_no_cache(mock_testinstance,
#                                  mock_testproblem,
#                                  mock_solverconfiguration,
#                                  string_testinstance,
#                                  string_testproblem,
#                                  string_solverconfiguration,
#                                  default_parameters):
#     # Set up mocks
#     mock_testinstance = string_testinstance
#     mock_testproblem = string_testproblem
#     mock_solverconfiguration = string_solverconfiguration

#     # Test without cache
#     framework1 = t.TestFramework(**default_parameters)
#     framework1.problems = ["prob1", "prob2"]
#     framework1.configs = ["config1", "config2"]

#     def init_side_effect(problem, config):
#         instance = MagicMock(name="testinstance_{}_{}".format(problem, config))
#         instance.run.return_value = (problem, config)
#         instance.__hash__.return_value = hash((problem, config))
#         instance.testproblem.id = problem
#         instance.config.id = config
#         return instance # a mocked instance of class TestInstance
#     mock_testinstance.side_effect = init_side_effect
#     framework1.solve_all_parallel(use_cache = False)


#     results = [("prob1", "config1"),
#                ("prob2", "config1"),
#                ("prob1", "config2"),
#                ("prob2", "config2")]
#     assert sorted(framework1.results) == sorted(results)


# @patch("cvxbenchmarks.framework.pkl.dump")
# @patch("cvxbenchmarks.framework.pkl.load")
# @patch("cvxbenchmarks.framework.SolverConfiguration")
# @patch("cvxbenchmarks.framework.TestProblem")
# @patch("cvxbenchmarks.framework.TestInstance")
# def test_testframework_solve_all_parallel_cache(mock_testinstance,
#                                  mock_testproblem,
#                                  mock_solverconfiguration,
#                                  mock_pickle_load,
#                                  mock_pickle_dump,
#                                  string_testinstance,
#                                  string_testproblem,
#                                  string_solverconfiguration,
#                                  default_parameters,
#                                  before_cache_pkl,
#                                  after_cache_pkl):
#     # Set up mocks
#     mock_testinstance = string_testinstance
#     mock_testproblem = string_testproblem
#     mock_solverconfiguration = string_solverconfiguration


#     from copy import deepcopy
#     old_cache = deepcopy(before_cache_pkl)
#     mock_pickle_load.return_value = before_cache_pkl
#     m = mock_open()

#     def init_side_effect(problem, config):
#         instance = MagicMock(name="testinstance_{}_{}".format(problem, config))
#         instance.run.return_value = (problem, config)
#         instance.__hash__.return_value = hash((problem, config))
#         instance.testproblem.id = problem
#         instance.config.id = config
#         return instance # a mocked instance of class TestInstance
#     mock_testinstance.side_effect = init_side_effect
#     framework2 = t.TestFramework(**default_parameters)
#     framework2.problems = ["prob1", "prob2"]
#     framework2.configs = ["config1", "config2"]
#     results = [("prob1", "config1"),
#                ("prob2", "config1"),
#                ("prob1", "config2"),
#                ("prob2", "config2")]



#     with patch("cvxbenchmarks.framework.open", m):
#         framework2.solve_all_parallel(use_cache = True)
#         assert sorted(framework2.results) == sorted(results)

#         # Make sure results were not run if they were cached.
#         for instance in framework2.instances:
#             if hash(instance) in old_cache:
#                 print(instance.run.mock_calls)

#         # Make sure the final cache contains all results.
#         assert call("cache.pkl", "wb") in m.mock_calls
#         # print(mock_pickle_dump.mock_calls)
#         # print(after_cache_pkl)
#         assert call(after_cache_pkl, m. return_value) in \
#             mock_pickle_dump.mock_calls

def test_testframework_export_results():
    pass









