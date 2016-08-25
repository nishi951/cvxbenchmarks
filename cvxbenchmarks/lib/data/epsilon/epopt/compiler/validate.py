

from epopt import error
from epopt.proto.epsilon.expression_pb2 import Expression, Problem

class ValidateError(error.ProblemError):
    pass


def check_sum_of_prox(problem):
    if problem.objective.expression_type != Expression.ADD:
        raise ValidateError("Objective is not ADD", problem)
