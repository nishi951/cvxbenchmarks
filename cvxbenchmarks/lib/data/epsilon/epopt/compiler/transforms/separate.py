"""Analyze the problem in sum-of-prox form and combine/split terms."""

from collections import defaultdict

from epopt import expression
from epopt import tree_format
from epopt.compiler import validate
from epopt.compiler.problem_graph import *
from epopt.compiler.transforms import linear
from epopt.compiler.transforms.transform_util import *
from epopt.proto.epsilon.expression_pb2 import Expression, ProxFunction
from epopt.util import *

def replace_var(expr, old_var_id, new_var):
    if (expr.expression_type == Expression.VARIABLE and
        expr.variable.variable_id == old_var_id):
        return new_var
    return expression.from_proto(
        expr.proto,
        [replace_var(arg, old_var_id, new_var) for arg in expr.arg],
        expr.data)

def is_least_squares_function(f):
    return f.expr.prox_function.prox_function_type in (
        ProxFunction.AFFINE,
        ProxFunction.CONSTANT,
        ProxFunction.SUM_SQUARE,
        ProxFunction.ZERO) and not f.expr.prox_function.epigraph

def separate_var(f_var):
    variable_id = "separate:%s:%s" % (
        f_var.variable, fp_expr(f_var.function.expr))
    return Expression(
        expression_type=Expression.VARIABLE,
        variable=Variable(variable_id=variable_id),
        size=f_var.instances[0].size)

def move_equality_indicators(graph):
    """Move certain equality indicators from objective to constraints."""
    # Single prox case, dont move it
    if len(graph.nodes(FUNCTION)) == 1:
        return

    for f in graph.nodes(FUNCTION):
        if f.expr.prox_function.prox_function_type == ProxFunction.ZERO:
            # Modify it to be an equality constraint
            f.expr = expression.indicator(Cone.ZERO, f.expr.arg[0])
            f.node_type = CONSTRAINT

def is_prox_friendly_constraint(expr, var_id):
    return expr.arg[0].affine_props.linear_maps[var_id].scalar

def has_incompatible_constraints(f, var, graph):
    if is_least_squares_function(f):
        return False

    var_id = var.expr.variable.variable_id
    for f in graph.neighbors(var, CONSTRAINT):
        if not is_prox_friendly_constraint(f.expr, var_id):
            return True

    return False

def add_variable_copy(f, var, graph):
    m, n = dims(var.expr)
    old_var_id = var.expr.variable.variable_id
    new_var_id = "separate:%s:%s" % (old_var_id, f.node_id)

    new_var = graph.add_node(
        expression.variable(m, n, new_var_id), VARIABLE, new_var_id)
    f.expr = replace_var(f.expr, old_var_id, new_var.expr)
    graph.remove_edge(f, var)
    graph.add_edge(f, new_var)

    eq_constr = graph.add_node(linear.transform_expr(
        expression.eq_constraint(new_var.expr, var.expr)), CONSTRAINT)
    graph.add_edge(eq_constr, new_var)
    graph.add_edge(eq_constr, var)

def separate_objective_terms(graph):
    for f in graph.nodes(FUNCTION):
        for var in graph.neighbors(f, VARIABLE):
            if (len(graph.neighbors(var, FUNCTION)) > 1 or
                has_incompatible_constraints(f, var, graph)):
                add_variable_copy(f, var, graph)

def add_constant_prox(graph):
    """Add f(x) = 0 term for variables only appearing in constraints."""

    for var in graph.nodes(VARIABLE):
        # Only add constant prox for variables not appearing in objective
        if graph.neighbors(var, FUNCTION):
            continue

        f_expr = expression.prox_function(
            ProxFunction(prox_function_type=ProxFunction.CONSTANT), var.expr)
        graph.add_edge(graph.add_node(f_expr, FUNCTION), var)

def variables(expr):
    if expr.expression_type == Expression.VARIABLE:
        yield expr
    for arg in expr.arg:
        for var in variables(arg):
            yield var

def add_function(f_expr, node_type, graph):
    var_list = list(variables(f_expr))

    # Exclude constant functions
    if not var_list:
        return

    f = graph.add_node(f_expr, node_type)
    for var_expr in var_list:
        var_id = var_expr.variable.variable_id
        graph.add_edge(f, graph.add_node(var_expr, VARIABLE, node_id=var_id))

def build_graph(problem):
    graph = ProblemGraph()
    for f_expr in problem.objective.arg:
        add_function(f_expr, FUNCTION, graph)
    for constr_expr in problem.constraint:
        add_function(constr_expr, CONSTRAINT, graph)
    return graph

GRAPH_TRANSFORMS = [
    move_equality_indicators,
    separate_objective_terms,
    add_constant_prox,
]

def transform_problem(problem, params):
    validate.check_sum_of_prox(problem)
    graph = build_graph(problem)

    if not graph.nodes(VARIABLE):
        return problem

    for f in GRAPH_TRANSFORMS:
        f(graph)
        log_debug(
            lambda f, graph:
            "%s:\n%s" %
            (f.__name__,
             tree_format.format_problem(graph.problem)),
            f, graph)
    return graph.problem
