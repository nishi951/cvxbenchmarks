# view
# See specific columns of the results table

from cvxbenchmarks.scripts.utils import get_command_line_args
import pandas as pd

def main(args):
    # Load results
    print("Loading results from {}...".format(args.results))
    results = pd.read_pickle(args.results)
    # Need to sort before accessing:
    results.sort_index(inplace=True)

    problemIndex = results.index.levels[0]
    configIndex = results.index.levels[1]
    statsIndex = results.columns
    if args.problems is not None:
        problemIndex = pd.Index(args.problems)
    if args.configs is not None:
        configIndex = pd.Index(args.configs)
    if args.stats is not None:
        statsIndex = pd.Index(args.stats)

    if not (args.include_nonoptimal):
        # Analyze only problems solved optimally by all solvers.
        results_statuses = results["status"].unstack()
        # Get Boolean dataframe of problems by configs
        all_opt = (results_statuses == "optimal").apply(all, axis=1)
        problemIndex = problemIndex & all_opt[all_opt].index # Get the problems solved optimally
        # results = results.loc[(all_opt_index, slice(None)),slice(None)]

    print(results.loc[(list(problemIndex), list(configIndex)), list(statsIndex)].to_string())

if __name__ == '__main__':
    main(get_command_line_args())