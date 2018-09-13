# settings.py
# Mark Nishimura 2017
# Manages various settings, global variables, strings, etc. for cvxbenchmarks

# CVXConfig
# Configuration file types supported:
YAML = "yaml" # Configuration file type

# CVXProblem
# Problem file types supported:
PY = "py"

# Problem Tags
LP = "LP"
QP = "QP"
SOCP = "SOCP"
SDP = "SDP"
EXP = "EXP"
MIP = "MIP"
TAGS = [LP, QP, SOCP, SDP, EXP, MIP]

# Keywords
PROBLEM_ID = "problemID"
CONFIG_ID = "configID"
INSTANCEHASH = "instancehash"
PROBLEM = "problem"
OPT_VAL = "opt_val"
TEST_PROBLEM_ID = "test_problem_id"
SOLVE_TIME = "solve_time"
SETUP_TIME = "setup_time"
NUM_ITERS = "num_iters"
STATUS = "status"
AVG_ABS_RESID = "avg_abs_resid"
MAX_RESID = "max_resid"

# From size_metrics



# Searching through problem files
PROBLEM_LIST = ["problems"]

# Cache Key for TestInstance hex digests:
TEST_INSTANCE_CACHE = "TEST_INSTANCE_CACHE"
RESULTS_CACHE = "RESULTS_CACHE"


