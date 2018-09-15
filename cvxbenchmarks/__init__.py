# cvxbenchmarks __init__.py

# Set environment variable path to root directory
import os
from os.path import realpath, dirname
CVXBENCH_DIR = realpath(dirname(__file__))
os.environ['CVXBENCH_DIR'] = CVXBENCH_DIR