

from epopt import expression_util
from epopt.compiler import validate
from epopt.proto.epsilon.expression_pb2 import Expression, Cone, LinearMap, ProxFunction

NAMES = {
    Expression.VARIABLE: "xyzwvutsrq",
    Expression.CONSTANT: "abcdkeflmn",
}

class NameMap(object):
    def __init__(self):
        self.name_map = {}
        self.count = {
            Expression.VARIABLE: 0,
            Expression.CONSTANT: 0,
        }

    def constant_name(self, constant):
        assert constant.data_location or constant.parameter_id
        return self.name(
            constant.data_location if constant.data_location else
            constant.parameter_id,
            Expression.CONSTANT,
            constant.n != 1)

    def variable_name(self, var_expr):
        assert var_expr.variable.variable_id
        return self.name(
            var_expr.variable.variable_id,
            Expression.VARIABLE,
            var_expr.size.dim[1] != 1)

    def name(self, name_id, name_type, is_matrix):
        if name_id in self.name_map:
            return self.name_map[name_id]

        name = NAMES[name_type][self.count[name_type] % len(NAMES[name_type])]
        if is_matrix:
            name = name.upper()

        self.name_map[name_id] = name
        self.count[name_type] += 1
        return name

def function_name(proto):
    if proto.expression_type == Expression.INDICATOR:
        return Cone.Type.Name(proto.cone.cone_type).lower()
    elif proto.expression_type == Expression.PROX_FUNCTION:
        return ProxFunction.Type.Name(
            proto.prox_function.prox_function_type).lower()
    return Expression.Type.Name(proto.expression_type).lower()

def format_params(proto):
    retval = []
    if proto.expression_type == Expression.INDEX:
        for key in proto.key:
            retval += ["%d:%d" % (key.start, key.stop)]
    elif proto.expression_type in (Expression.POWER, Expression.NORM_P):
        retval += [str(proto.p)]
    elif proto.expression_type == Expression.SUM_LARGEST:
        retval += [str(proto.k)]
    elif proto.expression_type == Expression.SCALED_ZONE:
        retval += ["alpha=%.2f" % proto.scaled_zone_params.alpha,
                   "beta=%.2f" % proto.scaled_zone_params.beta,
                   "C=%.2f" % proto.scaled_zone_params.c,
                   "M=%.2f" % proto.scaled_zone_params.m]

    if retval:
        return "[" + ", ".join(retval) + "]"
    else:
        return ""

def linear_map_name(linear_map, name_map):
    if linear_map.linear_map_type == LinearMap.DENSE_MATRIX:
        return "dense(" + name_map.constant_name(linear_map.constant) + ")"
    elif linear_map.linear_map_type == LinearMap.SPARSE_MATRIX:
        return "sparse(" + name_map.constant_name(linear_map.constant) + ")"
    elif linear_map.linear_map_type == LinearMap.DIAGONAL_MATRIX:
        return "diag(" + name_map.constant_name(linear_map.constant) + ")"
    elif linear_map.linear_map_type == LinearMap.SCALAR:
        return "scalar(%.2f)" % linear_map.scalar
    elif linear_map.linear_map_type == LinearMap.KRONECKER_PRODUCT:
        assert len(linear_map.arg) == 2
        return "kron(" + ", ".join(linear_map_name(arg, name_map)
                                   for arg in linear_map.arg) + ")"
    elif linear_map.linear_map_type == LinearMap.TRANSPOSE:
        assert len(linear_map.arg) == 1
        return "transpose(" + linear_map_name(linear_map.arg[0], name_map) + ")"

    raise ValueError("unknown linear map type: %d" % linear_map.linear_map_type)

def format_linear_map(expr, name_map):
    assert len(expr.arg) == 1
    return (linear_map_name(expr.linear_map, name_map) + "*" +
            format_expr(expr.arg[0], name_map))

def format_expr(expr, name_map):
    if expr.expression_type == Expression.CONSTANT:
        if not expr.constant.data_location:
            return "%.2f" % expr.constant.scalar
        return "const(" + name_map.constant_name(expr.constant) + ")"
    elif expr.expression_type == Expression.VARIABLE:
        return "var(" + name_map.variable_name(expr) + ")"
    elif expr.expression_type == Expression.LINEAR_MAP:
        return format_linear_map(expr, name_map)
    elif expr.expression_type == Expression.RESHAPE:
        assert len(expr.arg) == 1
        return format_expr(expr.arg[0], name_map)

    return (function_name(expr) + format_params(expr) +
            "(" + ", ".join(format_expr(arg, name_map)
                            for arg in expr.arg) + ")")

def format_problem(problem):
    name_map = NameMap()
    validate.check_sum_of_prox(problem)

    output = "objective:\n"
    output += ("  add(\n    " +
               ",\n    ".join(
                   format_expr(arg, name_map) for arg in problem.objective.arg) +
               ")\n")

    if problem.constraint:
        output += "\nconstraints:\n"
        output += "".join("  " + format_expr(constr, name_map) + "\n"
                          for constr in problem.constraint)

    return output
