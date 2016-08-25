"""Visualize expression trees using networkx"""

import pygraphviz as pgv

from epopt import expression_str
from epopt import expression_util
from epopt.proto.epsilon import expression_pb2

def node_label(expr):
    return (expression_pb2.Expression.Type.Name(expr.expression_type) + " " +
            expression_str.node_contents_str(expr))

def add_graph_node(expr, parent_fp, G):
    expr_fp = expression_util.fp_expr(expr)
    G.add_node(expr_fp, label=node_label(expr))

    if parent_fp:
        G.add_edge(parent_fp, expr_fp)

    for arg in expr.arg:
        add_graph_node(arg, expr_fp, G)

def graph(expr):
    G = pgv.AGraph(directed=True)
    add_graph_node(expr, None, G)
    return G
