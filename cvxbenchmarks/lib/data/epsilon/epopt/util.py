
import logging
import resource

from epopt import tree_format

def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF).ru_utime

class DeferredMessage(object):
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def __str__(self):
        return self.func(*self.args)

def log_debug(f, *args):
    logging.debug(DeferredMessage(f, *args))

def log_debug_expr(msg, expr):
    log_debug(lambda msg, expr: msg + ":\n" + tree_format.format_expr(expr),
              msg, expr)
