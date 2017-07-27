# settings.py
# Mark Nishimura 2017
# Manages various settings, global variables, strings, etc. for cvxbenchmarks

# Problem Tags
LP = "LP"
QP = "QP"
SOCP = "SOCP"
SDP = "SDP"
EXP = "EXP"
MIP = "MIP"
TAGS = [LP, QP, SOCP, SDP, EXP, MIP]

# TestProblem Parameters
PROBLEM_ID = "problemID"
PROBLEM = "problem"
OPT_VAL = "opt_val"

# Searching through problem files
PROBLEM_LIST = ["problems"]

# Cache Key for TestInstance hex digests:
TEST_INSTANCE_CACHE = "TEST_INSTANCE_CACHE"
RESULTS_CACHE = "RESULTS_CACHE"


