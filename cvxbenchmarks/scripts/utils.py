# utils.py
# Various utility functions.

import argparse

def get_command_line_args():
    """
    TODO: Implement all options:

    analyze : Produce and display results from running the tests
        options:
            -r, --results : the file where the results are stored 
            (defaults to results.dat)

    generate :  
        options:
            --tags : A list of tags of problems to generate
            --templateDir : the directory containing the problem 
                            templates (.j2)
            --paramDir : the directory containing the parameter 
                         files (.csv)
            --template-and-params: specify a particular template to generate 
                                   and a particular params file to use. If 
                                   there are multiple parameter settings, it 
                                   will generate all of them.
            --problemDir : the directory containing the problems (where the 
                           problems will be written)
    run
        options:
            --no-cache : (boolean) Turn off cache usage.
            --clear-cache : Clear the cache.
            --problem : specify a particular problem to benchmark. also takes 
                        a list of problems
            --problemDir : the directory containing the problems (from which 
                           the problems will be read)

    """
    parser = argparse.ArgumentParser("cvxbench")
    parser.add_argument("script", 
                         type=str, 
                         choices=("analyze", 
                                  "generate",
                                  "run",
                                  "render"),
                         help="the cvxbench script to run.")

    # Script-specific
    # analyze
    parser.add_argument("--results", type=str, 
                        default="cvxbenchmarks/results.dat",
                        help=("the data file to which the results data " +
                              "frame will be written."))

    # generate
    parser.add_argument("--tags", type=str, nargs='+',
                        help=("a list of tags that determine which problems " + 
                              "are run."))
    parser.add_argument("--templateDir", type=str, 
                        default="cvxbenchmarks/lib/data/cvxbenchmarks",
                        help=("the directory containing the template files."))
    parser.add_argument("--paramDir", type=str, 
                        default="cvxbenchmarks/lib/data/cvxbenchmarks",
                        help=("the directory containing the parameter files."))
    parser.add_argument("--template-and-params", type=str, nargs=2,
                        help="specify a template file and a parameter file.")
    parser.add_argument("--problemDir", type=str, 
                        default="cvxbenchmarks/problems",
                        help=("the directory containing the (.py) written " +
                        "problems.")) # Also used in run
    parser.add_argument("--use-index", action="store_true",
                        help="whether or not to use the index object.")

    # run
    # --problemDir
    parser.add_argument("--no-cache", action='store_true',
                        help="disable cache use during run.") # Double negatives...
    parser.add_argument("--clear-cache", action='store_true',
                        help="clear the problem results cache.")
    parser.add_argument("--problems", type=str, 
                        help="specify a list of problems to benchmark.")
    parser.add_argument("--configDir", type=str, 
                        default="cvxbenchmarks/lib/configs",
                        help="the directory containing the (.py) " +
                        "configuration files.")
    parser.add_argument("--configs", type=str, nargs='+',
                        help=("a list of <solver>_config configurations. "+
                        "if unset, the framework runs all available "+
                        "configurations."))

    # render
    # --template-and-params

    args = parser.parse_args()
    print(args)
    return args