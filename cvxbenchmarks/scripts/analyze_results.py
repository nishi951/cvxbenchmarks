# analyze_results.py
# Mark Nishimura 2017

import pandas as pd
import numpy as np
import cvxbenchmarks.data_visualization as dv

import math

from cvxbenchmarks.scripts.utils import get_command_line_args

def main(args):
    # Load results.dat
    print("Loading results from {}...".format(args.results))
    results = pd.read_pickle(args.results)
    # results = pd.read_hdf(args.results)
    print(results.to_string())
    print("Done.")

    # Sort the dataframe in place before accessing it by index:
    results.sort_index(inplace=True)

    if args.problems is None:
        problemIndex = results.axes[0].levels[0]
    else:
        problemIndex = pd.Index(args.problems)

    if args.configs is None:
        configIndex = results.axes[0].levels[1]
    else:
        configIndex = pd.Index(args.configs)
    labels = results.axes[1]

    if not (args.include_nonoptimal):
        # Analyze only problems solved optimally by all solvers.
        results_statuses = results["status"].unstack()
        # Get Boolean dataframe of problems by configs
        all_opt = (results_statuses == "optimal").apply(all, axis=1)
        problemIndex = problemIndex & all_opt[all_opt].index # Get the problems solved optimally
        # results = results.loc[(all_opt_index, slice(None)),slice(None)]

    # Save solve time latex
    with open("time_table.tex", "w") as f:
        f.write(results.loc[(slice(None), configIndex), "solve_time"].fillna("-").to_latex())

    # Data Visualization
    # Generate performance profiles for all solver configurations
    # Graph performance profile:
    import matplotlib.pyplot as plt

    plt.figure()
    dv.plot_performance_profile(results,
                                    problemIndex=problemIndex,
                                    configIndex=configIndex,
                                    xmax=40)
    plt.draw()

    # Graph time vs. big(small)^2
    # plt.figure()
    # dv.plot_scatter_by_config(results, "max_big_small_squared", "solve_time")
    # plt.draw()

    # Graph time vs. number of scalar variables
    plt.figure()
    dv.plot_scatter_by_config(results, "num_scalar_variables", "solve_time", 
                                  problemIndex=problemIndex, 
                                  configIndex=configIndex, 
                                  logx=True, 
                                  logy=True)
    plt.draw()

    # Graph time vs. number of scalar data
    plt.figure()
    dv.plot_scatter_by_config(results, "num_scalar_data", "solve_time", 
                                  problemIndex=problemIndex, 
                                  configIndex=configIndex,
                                  logx=True, 
                                  logy=True)
    plt.draw()

    # Graph time vs. number of scalar constraints
    plt.figure()
    dv.plot_scatter_by_config(results, ["num_scalar_eq_constr", "num_scalar_leq_constr"], "solve_time", 
                                  problemIndex=problemIndex,
                                  configIndex=configIndex,
                                  logx=False,
                                  logy=True)
    plt.draw()

    # Graph num_iterations vs. number of scalar variables
    # Graph time vs. number of scalar variables
    plt.figure()
    dv.plot_scatter_by_config(results, "num_scalar_variables", "num_iters", 
                                  problemIndex=problemIndex,
                                  configIndex=configIndex,
                                  logx=True,
                                  logy=False)
    plt.draw()

    # Graph histogram of solve accuracies (relative to mosek)
    # plt.figure()
    # dv.plot_histograms_by_config(results)
    # plt.draw()

    # Show figures
    # plt.show()

    # Save figures to a single file:
    if args.format == "pdf":
        # http://stackoverflow.com/questions/26368876/saving-all-open-matplotlib-figures-in-one-file-at-once
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        def multipage(filename, figs=None, dpi=200):
            pp = PdfPages(filename)
            if figs is None:
                figs = [plt.figure(n) for n in plt.get_fignums()]
            for fig in figs:
                fig.savefig(pp, format='pdf')
            pp.close()

        multipage("figs.pdf")
        with pd.option_context('display.max_rows', None):
            print(results.to_frame(filter_observations = False))
        # items = statistics
        # major_axis = problemIDs
        # minor_axis = configIDs
    else:
        for i in plt.get_fignums():
            plt.figure(i)
            plt.tight_layout()
            plt.savefig('figure{}.png'.format(i))

    # Find infeasible problems:
    for problem in problemIndex:
        for config in configIndex:
            if results.loc[(problem, config), "status"] == "infeasible":
                print(str((problem, config)) + " gives infeasible.")
            if results.loc[(problem, config), "status"] is None:
                print(str((problem, config)) + " gives None.")

    # Solver statuses:
    status_counts = pd.DataFrame(0, index = configIndex, columns = ["optimal", "optimal_inaccurate", "infeasible", "unbounded","nan"])
    for config in configIndex:
        for problem in problemIndex:
            status_counts.loc[config, str(results.loc[(problem, config), "status"])] += 1
    print("Solver status counts")
    print(status_counts)

    # Calculate aggregate statistics:
    # Average error:
    if "error" in results.columns:
        avg_errors = pd.Series(index = configIndex)
        for config in configIndex:
            total_error = 0
            num_not_null = 0.0
            for problem in problemIndex:
                if pd.notnull(results.loc[(problem, config), "error"]) and \
                  results.loc[(problem, config), "status"] == "optimal":
                    num_not_null += 1
                    total_error += results.loc[(problem, config), "error"]
            if num_not_null > 0:
                avg_errors.loc[config] = total_error / num_not_null
            else:
                avg_errors.loc[config] = None

        print("Average errors:")
        print(avg_errors)

    # Relative performance:
    # For each pair of configs:
    rel_performance = pd.DataFrame(index = configIndex, columns = configIndex)
    for standard in rel_performance.index:
        for compare in rel_performance.columns:
            # Compute average ratio (compare/standard) runtimes for problems that 
            # were solved by both configurations
            num_solved_by_both = 0
            total_compare = 0.0
            total_standard = 0.0
            for problem in problemIndex:
                standard_time = results.loc[(problem, standard), "solve_time"]
                compare_time = results.loc[(problem, compare), "solve_time"]
                if pd.notnull(standard_time) and pd.notnull(compare_time):
                    num_solved_by_both += 1
                    total_compare += compare_time
                    total_standard += standard_time
            if num_solved_by_both > 0:
                rel_performance.loc[standard, compare] = total_compare/total_standard
            else:
                rel_performance.loc[standard, compare] = None
    print("Relative performances")
    print(rel_performance)

    # Number of iterations
    avg_num_iters = pd.Series(index = configIndex)
    for config in configIndex:
        total_num_iters = 0
        num_not_null = 0.0
        for problem in problemIndex:
            if pd.notnull(results.loc[(problem, config), "num_iters"]) and \
               results.loc[(problem, config), "status"] == "optimal":
                num_not_null += 1
                total_num_iters += results.loc[(problem, config), "num_iters"]
        if num_not_null > 0:
            avg_num_iters.loc[config] = total_num_iters / num_not_null
        else:
            avg_num_iters.loc[config] = None
    print("Average number of iterations")
    print(avg_num_iters)

    # Linear regression (scaling):
    # https://stackoverflow.com/questions/19991445/run-an-ols-regression-with-pandas-data-frame
    # import matplotlib.pyplot as plt
    # import statsmodels.formula.api as sm

    # print("solve_time vs. num_scalar_variables")
    # print("-----------------------------------")

    # for config in configIndex:
    #     configData = results.minor_xs(config) # returns a DataFrame
    #     nvars = configData.loc[:, "num_scalar_variables"]
    #     solvetimes = configData.loc[:, "solve_time"]
    #     df = pd.DataFrame({"X" : nvars, "Y" : solvetimes})
    #     ls_reg = sm.ols(formula="X ~ Y", data=df).fit()
    #     print("config: " + config)
    #     print(ls_reg.params)

    # return


if __name__ == '__main__':
    main(get_command_line_args())  




