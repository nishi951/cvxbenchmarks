import logging

from epopt import tree_format
from epopt.compiler.transforms import prox
from epopt.compiler.transforms import separate
from epopt.compiler.transforms import split
from epopt.proto.epsilon import solver_params_pb2

# TODO(mwytock): Add this back
# split.transform_problem,

TRANSFORMS = [
    prox.transform_problem,
    separate.transform_problem,
]

def transform_name(transform):
    return ".".join((transform.__module__, transform.__name__))

def compile_problem(problem, params=solver_params_pb2.SolverParams()):
    logging.debug("params:\n%s", params)
    logging.debug("input:\n%s", tree_format.format_problem(problem))
    for transform in TRANSFORMS:
        problem = transform(problem, params)
        logging.debug(
            "%s:\n%s",
            transform_name(transform),
            tree_format.format_problem(problem))
    return problem
