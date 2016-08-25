
import numpy as np
import scipy.sparse as sp

from epopt.proto.epsilon.expression_pb2 import Constant

def value_location(value):
    return "/mem/data/" + str(abs(hash(value)))

def value_data(value):
    # TODO(mwytock): Need to ensure type double/float here?
    if isinstance(value, np.ndarray):
        constant = Constant(
            constant_type=Constant.DENSE_MATRIX,
            m=value.shape[0],
            n=1 if len(value.shape) == 1 else value.shape[1])
        value_bytes = value.tobytes(order="F")

    elif isinstance(value, sp.spmatrix):
        csc = value.tocsc()
        constant = Constant(
            constant_type=Constant.SPARSE_MATRIX,
            m=value.shape[0],
            n=1 if len(value.shape) == 1 else value.shape[1],
            nnz=value.nnz)

        value_bytes = (csc.indptr.tobytes("F") +
                       csc.indices.tobytes("F") +
                       csc.data.tobytes("F"))

    else:
        raise ValueError("unknown value type " + str(value))

    return constant, value_bytes

def store(value, data_map):
    constant, value_bytes = value_data(value)
    location = value_location(value_bytes)
    data_map[location] = value_bytes
    constant.data_location = location
    return constant
