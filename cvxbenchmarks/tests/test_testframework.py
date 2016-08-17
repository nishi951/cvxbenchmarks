
import cvxbenchmarks.TestFramework
from cvxbenchmarks.tests.base_test import BaseTest
import sys, os, inspect

TESTS_PROBLEM_DIR = "/Users/mark/Documents/Stanford/reu2016/project/src/tests/problems"
TESTS_CONFIG_DIR = "/Users/mark/Documents/Stanford/reu2016/project/src/tests/configs"


class Test_testFramework(BaseTest):
    """Unit tests for the TestFramework class
    """
    def setUp(self):
        self.framework = src.TestFramework.TestFramework(TESTS_PROBLEM_DIR, TESTS_CONFIG_DIR, problems = [], configs = [])
        sys.path.insert(0, TESTS_PROBLEM_DIR)
        sys.path.insert(0, TESTS_CONFIG_DIR)
        # print "setup!"

    def test_load_problem(self):
        """Tests the load_problem method.
        """
        self.framework.load_problem("lp_0")
        ref = src.TestFramework.TestProblem("lp_0", __import__("lp_0").prob)
        self.assertEqual(self.framework.problems[0].problem, ref.problem)
        self.assertEqual(self.framework.problems[0].id, ref.id)

    def test_preload_all_problems(self):
        """Tests the preload_all_problems method.
        """
        self.framework.preload_all_problems()
        least_squares_0 = src.TestFramework.TestProblem("least_squares_0", __import__("least_squares_0").prob)
        lp_0 = src.TestFramework.TestProblem("lp_0", __import__("lp_0").prob)
        self.assertEqual(self.framework.problems[0], least_squares_0)
        self.assertEqual(self.framework.problems[1], lp_0)

    def test_load_config(self):
        """Tests the load_config method.
        """
        self.framework.load_config("mosek_config")
        mosek_config = __import__("mosek_config")
        ref_solver = mosek_config.solver
        ref_verbose = mosek_config.verbose
        ref_kwargs = mosek_config.kwargs
        self.assertEqual(self.framework.configs[0].solver, ref_solver)
        self.assertEqual(self.framework.configs[0].verbose, ref_verbose)
        self.assertDictEqual(self.framework.configs[0].kwargs, ref_kwargs)

    def test_preload_all_configs(self):
        """Tests the preload_all_configs method.
        """
        self.framework.preload_all_configs()
        ecos_config = __import__("ecos_config")
        self.assertEqual(self.framework.configs[0].solver, ecos_config.solver)
        self.assertEqual(self.framework.configs[0].verbose, ecos_config.verbose)
        self.assertDictEqual(self.framework.configs[0].kwargs, ecos_config.kwargs)

        mosek_config = __import__("mosek_config")
        self.assertEqual(self.framework.configs[1].solver, mosek_config.solver)
        self.assertEqual(self.framework.configs[1].verbose, mosek_config.verbose)
        self.assertDictEqual(self.framework.configs[1].kwargs, mosek_config.kwargs)

    def test_generate_test_instances(self):
        """Tests the generate_test_instances method.
        """
        lp_0 = src.TestFramework.TestProblem("lp_0", __import__("lp_0").prob)
        mosek = __import__("mosek_config")
        mosek_config = src.TestFramework.SolverConfiguration("mosek_config", mosek.solver, mosek.verbose, mosek.kwargs)
        ref = src.TestFramework.TestInstance(lp_0, mosek_config)

        self.framework = src.TestFramework.TestFramework(TESTS_PROBLEM_DIR, TESTS_CONFIG_DIR, 
            problems = [lp_0], configs = [mosek_config])
        self.framework.generate_test_instances()
        # Compare testproblem
        self.assertEqual(self.framework._instances[0].testproblem.id, lp_0.id)
        self.assertEqual(self.framework._instances[0].testproblem.problem, lp_0.problem)

        # Compare config
        self.assertEqual(self.framework._instances[0].config.solver, ref.config.solver)
        self.assertEqual(self.framework._instances[0].config.verbose, ref.config.verbose)
        self.assertDictEqual(self.framework._instances[0].config.kwargs, ref.config.kwargs)

    def test_solve_all(self):
        """Tests the solve_all method.
        """
        pass # TODO

    def test_solve_all_parallel(self):
        """Tests the solve_all_parallel method.
        """
        pass # TODO

        









