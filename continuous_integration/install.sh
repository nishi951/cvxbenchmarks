#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

# Deactivate travis virtualenv
deactivate

# Install Miniconda
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda2/bin:$PATH
conda update --yes conda

# Install scipy/numpy with mkl bindings
conda create -n testenv --yes python=$PYTHON_VERSION mkl pip  \
      numpy scipy ecos

source activate testenv
conda install -c cvxgrp --yes scs multiprocess cvxcanon
# Reinstall numpy to activate mkl
# https://github.com/ContinuumIO/anaconda-issues/issues/720
conda install -f numpy --yes



# if [[ "$DISTRIB" == "ubuntu" ]]; then
#     sudo apt-get update -qq
#     # Use standard ubuntu packages in their default version
#     sudo apt-get install -qq python-pip python-scipy python-numpy
# fi

# if [[ "$COVERAGE" == "true" ]]; then
#     pip install coverage coveralls
# fi

# Install cvxbenchmarks
python setup.py install
