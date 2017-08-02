# cvxbenchmarks
A testing framework for convex solvers in CVXPY.

## Installation

After cloning the repo, run:
~~~
$ cd cvxbenchmarks
$ python setup.py install
~~~

If you have pytest installed (e.g. by running `pip install pytest`) You can check your installation by running:
~~~
$ py.test
~~~

## Example
`cvxpy` comes with the [ECOS](https://github.com/embotech/ecos "ECOS") and [SCS](https://github.com/cvxgrp/scs "SCS")
solvers already installed. To see how they compare on some small sample problems, navigate to the `cvxbenchmarks` directory and run
~~~
$ cvxbench generate --example
$ cvxbench run --example
~~~
A pandas dataframe with relevant timing and size information should appear, and a `figs.pdf` file 
containing some useful plots should also be generated.

## Problems
The cvxbenchmarks problem database has a wide variety of parametrizable problem types. To see the types of problems available, check out the `cvxbenchmarks/examples` directory, which contains a variety of html files generated using `ipython nbviewer`. The original ipython notebooks can be found in `cvxbenchmarks/examples/ipython`.

In addition to the cvxbenchmarks problem database, it is possible to download problems from a number of other places for use with cvxbenchmarks.

Currently supported:
qpoases

In the works:
    - CUTest
    - DIMACS
    - CVXCanon

### Adding your own problems
TODO


Cvxbenchmarks was designed and implemented by Mark Nishimura with the 
input of Steven Diamond and Stephen Boyd.
