
import errno
import os
import resource

from epopt import constant
from epopt import cvxpy_expr
from epopt.compiler import compiler
from epopt.proto.epsilon.expression_pb2 import Expression

def modify_data_location_linear_map(linear_map, f):
    if linear_map.constant.data_location != "":
        linear_map.constant.data_location = f(linear_map.constant.data_location)

    for arg in linear_map.arg:
        modify_data_location_linear_map(arg, f)

def modify_data_location(expr, f):
    if (expr.expression_type == Expression.CONSTANT and
        expr.constant.data_location != ""):
        expr.constant.data_location = f(expr.constant.data_location)

    if expr.expression_type == Expression.LINEAR_MAP:
        modify_data_location_linear_map(expr.linear_map, f)

    if (expr.expression_type == Expression.PROX_FUNCTION and
        expr.prox_function.HasField("scaled_zone_params")):
        params = expr.prox_function.scaled_zone_params
        modify_data_location(params.alpha_expr, f)
        modify_data_location(params.beta_expr, f)

    for arg in expr.arg:
        modify_data_location(arg, f)

def makedirs_existok(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def write_problem(cvxpy_prob, location, name):
    """Utility function to write problems for analysis."""

    mem_prefix = "/mem/"
    file_prefix = "/local" + location + "/"
    def rewrite_location(name):
        assert name[:len(mem_prefix)] == mem_prefix
        return file_prefix + name[len(mem_prefix):]

    makedirs_existok(location)
    prob_proto = cvxpy_expr.convert_problem(cvxpy_prob)
    prob_proto = compiler.compile_problem(prob_proto)

    modify_data_location(prob_proto.objective, rewrite_location)
    for constraint in prob_proto.constraint:
        modify_data_location(constraint, rewrite_location)

    with open(os.path.join(location, name), "w") as f:
        f.write(prob_proto.SerializeToString())

    for name, value in constant.global_data_map.items():
        assert name[:len(mem_prefix)] == mem_prefix
        filename = os.path.join(location, name[len(mem_prefix):])
        makedirs_existok(os.path.dirname(filename))
        with open(filename, "w") as f:
            f.write(value)

def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF).ru_utime
