
from epopt import tree_format

from nose.tools import assert_equal

def assert_expr_equal(a, b):
    assert_equal(
        a, b,
        "\nExpected:\n" + tree_format.format_expr(a) +
        "\n!=\nActual:\n" + tree_format.format_expr(b))
