

from epopt.proto.epsilon.expression_pb2 import Expression

def name(proto):
    return Expression.Type.Name(proto.expression_type).lower()

def params(proto):
    retval = []

    if proto.expression_type == Expression.CONSTANT:
        if proto.constant.data_location:
            retval += ["data_location", proto.constant.data_location]
        if not proto.constant.data_location:
            retval += ["scalar", proto.constant.scalar]
    elif proto.expression_type == Expression.VARIABLE:
        retval += ["variable_id", proto.variable.variable_id]
    elif proto.expression_type == Expression.INDEX:
        retval += ["start", proto.key.start]
        retval += ["stop", proto.key.stop]
        retval += ["step",  proto.key.step]
    elif proto.expression_type in (Expression.POWER, Expression.NORM_P):
        retval += ["p", proto.p]
    elif proto.expression_type == Expression.SUM_LARGEST:
        retval += ["k", proto.k]
    elif proto.expression_type == Expression.INDICATOR:
        retval += ["cone", Cone.Type.Name(proto.cone.cone_type)]

    return retval

def expression(proto):
    return [name(proto), params(proto), [expression(arg) for arg in proto.arg]]

def format(proto):
    return ["problem",
            expression(proto.objective),
            [expression(constr) for constr in proto.constraint]]
