# cvxbenchmarks

TODO(mark) Update all of this once things settle down.

A testing framework for convex solvers in CVXPY.

## Setup
Cvxbenchmarks has currently only been tested on Python 2.7. (Python 3 coming soon). 
The [Anaconda](https://www.continuum.io/downloads "Anaconda Download Page") distribution is currently the
preferred way to install Python for cvxbenchmarks.

Cvxbenchmarks currently requires:
~~~
cvxpy >= 0.4.2
pandas >= 0.18.1
tabulate >= 0.7.5
~~~

After cloning the repo, run `python setup.py install` to make sure the relevant dependencies are installed.

## Example
Cvxpy comes with the [ECOS](https://github.com/embotech/ecos "ECOS") and [SCS](https://github.com/cvxgrp/scs "SCS")
solvers already installed. To see how they compare on some small sample problems, navigate to the `cvxbenchmarks/cvxbenchmarks` directory and run:
~~~
$ python generate_example.py
$ python run_example.py
~~~
A pandas dataframe with relevant timing and size information should appear, and a `figs.pdf` file 
containing some useful plots should also be generated.

Cvxbenchmarks was designed and implemented by Mark Nishimura with the 
input of Steven Diamond and Stephen Boyd.
