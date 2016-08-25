#!/usr/bin/env python
#
# Usage:
#   python -m epsilon.problems.print lasso '{"m":10 "n":5}'

import argparse
import json

from epopt import cvxpy_expr
from epopt import text_format
from epopt import tree_format
from epopt.compiler import compiler
from epopt.problems import *

FORMATTERS = {
    "text": text_format.format_problem,
    "tree": tree_format.format_problem,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("kwargs", help="Problem arg, e.g. {\"m\": 10}")
    parser.add_argument("--format", default="text")
    args = parser.parse_args()

    formatter = FORMATTERS[args.format]

    cvxpy_prob = locals()[args.problem].create(**json.loads(args.kwargs))
    problem = cvxpy_expr.convert_problem(cvxpy_prob)

    print "original:"
    print formatter(problem)

    for transform in compiler.TRANSFORMS:
        problem = transform(problem)

        print
        print ".".join((transform.__module__, transform.__name__)) + ":"
        print formatter(problem)
