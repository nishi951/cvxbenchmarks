from cvxbenchmarks.framework import TestFramework
import time
import pandas as pd

from cvxbenchmarks.scripts.utils import get_command_line_args

# NOTE: Do not import matplotlib.pyplot before running TestFramework's solve_all_parallel - 
# Something about matplotlib.pyplot messes up the solve (at least in a virtualenv)

def main(args):
    problemDir = args.problemDir
    configDir = args.configDir
    configs = args.configs
    problems = args.problems
    parallel = args.parallel


    framework = TestFramework(problemDir = problemDir, configDir = configDir)
    print("Loading problems...")
    if problems is None:
        framework.preload_all_problems()
    else:
        for problem in problems:
            framework.load

    print("\tDone.")

    print("Loading configs...")
    if configs is None:
        framework.preload_all_configs()
    else:
        for config in args.configs:
            framework.load_config(config)
    print("\tDone.")

    start = time.time()
    print("Solving all problem instances...")
    if parallel:
        framework.solve_all_parallel()
    else:
        framework.solve_all()
    print("\tDone.")


    # print("Solving all problem instances in parallel...")
    # framework.solve_all_parallel()
    # print("\tDone.")
    print("\tTime:",str(time.time()-start))

    print("number of results:", str(len(framework.results)))

    # Export results to a pandas panel
    print("exporting results.")
    results = framework.export_results_as_panel()
    print(results.to_frame(filter_observations = False)) #filter_observations = False prevents rows with NaN from not appearing.
    # Save data frame to a file.
    results.to_pickle("results.dat")

if __name__ == '__main__':
    main(get_command_line_args())






        




