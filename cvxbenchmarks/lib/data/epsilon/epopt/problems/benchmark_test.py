
import argparse

from epopt.problems import benchmark
from epopt.problems import lasso
from epopt.problems.problem_instance import ProblemInstance


def test_benchmarks():
    benchmark.run_benchmarks(
        [lambda p: 0],
        [ProblemInstance("lasso", lasso.create, dict(m=5, n=10))])
