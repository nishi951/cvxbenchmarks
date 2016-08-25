"""Represents the bipartite graph between functions/variables.

Functions can either be objective terms or constraints and edges represent the
usage of a variable by a function constraint.

We wrap the expression tree form from expression_pb2 with thin Python objects
that allow for caching of computations and object-oriented model for
mutations.
"""

from collections import defaultdict

from epopt import expression
from epopt import expression_util
from epopt import tree_format
from epopt.compiler import validate
from epopt.proto.epsilon.expression_pb2 import Expression, Problem, Cone

VARIABLE = "variable"
FUNCTION = "function"
CONSTRAINT = "constraint"

class Node(object):
    def __init__(self, expr, node_type, node_id):
        self.expr = expr
        self.node_type = node_type
        self.node_id = node_id

class ProblemGraph(object):
    def __init__(self):
        # dict stores count for ordering purposes
        self.node_values = {}
        self.edges = defaultdict(set)

    @property
    def problem(self):
        return expression.Problem(
            objective=expression.add(*(f.expr for f in self.nodes(FUNCTION))),
            constraint=[f.expr for f in self.nodes(CONSTRAINT)])

    # Basic operations
    def add_edge(self, a, b):
        assert a.node_id in self.node_values
        assert b.node_id in self.node_values
        self.edges[a.node_id].add(b.node_id)
        self.edges[b.node_id].add(a.node_id)

    def remove_edge(self, a, b):
        self.edges[a.node_id].remove(b.node_id)
        self.edges[b.node_id].remove(a.node_id)

    def add_node(self, expr, node_type, node_id=None):
        if node_id in self.node_values:
            return self.node_values[node_id][1]

        if node_id is None:
            node_id = node_type + str(len(self.node_values))
        node = Node(expr, node_type, node_id)
        self.node_values[node_id] = (len(self.node_values), node)
        return node

    # TODO(mwytock): Add remove_node()

    def nodes(self, node_type):
        return [n for idx, n in sorted(self.node_values.values())
                if n.node_type == node_type]

    def neighbors(self, a, node_type):
        return [n for idx, n in sorted(
            [self.node_values[b_id] for b_id in self.edges[a.node_id]
             if self.node_values[b_id][1].node_type == node_type])]
