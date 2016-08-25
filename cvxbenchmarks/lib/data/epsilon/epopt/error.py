from epopt.tree_format import *

class ProblemError(Exception):
    def __init__(self, message, problem):
        super(ProblemError, self).__init__(message)
        self.problem = problem

    def __str__(self):
        return (super(ProblemError, self).__str__() + "\n" +
                format_problem(self.problem))

class ExpressionError(Exception):
    def __init__(self, message, *expr_args):
        super(ExpressionError, self).__init__(message)
        self.expr_args = expr_args

    def __str__(self):
        return (super(ExpressionError, self).__str__() + "\n" +
                "\n\n".join(format_expr(expr) for expr in self.expr_args))

class LinearMapError(Exception):
    def __init__(self, message, *linear_map_args):
        super(LinearMapError, self).__init__(message)
        self.linear_map_args = linear_map_args

    def __str__(self):
        return (super(LinearMapError, self).__str__() + "\nLinear maps:\n" +
                "\n".join(format_linear_map(A) for A in self.linear_map_args))
