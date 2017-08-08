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
    print(results.to_frame(filter_observations = False).to_string())
    print("Done.")

    # Save solve time latex
    with open("time_table.tex", "w") as f:
        f.write(results.loc["solve_time", :, ["superscs_config", "scs_config"]].to_latex())


    # Data Visualization
    # Generate performance profiles for all solver configurations
    # Graph performance profile:
    import matplotlib.pyplot as plt

    plt.figure()
    dv.plot_performance_profile(results)
    plt.draw()

    # Graph time vs. big(small)^2
    plt.figure()
    dv.plot_scatter_by_config(results, "max_big_small_squared", "solve_time")
    plt.draw()

    # Graph time vs. number of scalar variables
    plt.figure()
    dv.plot_scatter_by_config(results, "num_scalar_variables", "solve_time", logx = True, logy = True)
    plt.draw()

    # Graph time vs. number of scalar data
    plt.figure()
    dv.plot_scatter_by_config(results, "num_scalar_data", "solve_time", logx = True, logy = True)
    plt.draw()

    # Graph time vs. number of scalar constraints
    plt.figure()
    dv.plot_scatter_by_config(results, ["num_scalar_eq_constr", "num_scalar_leq_constr"], "solve_time", logx = False, logy = True)
    plt.draw()

    # Graph num_iterations vs. number of scalar variables
    # Graph time vs. number of scalar variables
    plt.figure()
    dv.plot_scatter_by_config(results, "num_scalar_variables", "num_iters", logx = True, logy = False)
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
    for problem in results.major_axis:
        for config in results.minor_axis:
            if results.loc["status", problem, config] == "infeasible":
                print(str((problem, config)) + " gives infeasible.")
            if results.loc["status", problem, config] is None:
                print(str((problem, config)) + " gives None.")

    # Solver statuses:
    status_counts = pd.DataFrame(0, index = results.minor_axis, columns = ["optimal", "optimal_inaccurate", "infeasible", "None"])
    for config in results.minor_axis:
        for problem in results.major_axis:
            status_counts.loc[config, str(results.loc["status", problem, config])] += 1
    print("Solver status counts")
    print(status_counts)

    # Calculate aggregate statistics:
    # Average error:
    if "error" in results.items:
        avg_errors = pd.Series(index = results.minor_axis)
        for config in results.minor_axis:
            total_error = 0
            num_not_null = 0.0
            for problem in results.major_axis:
                if pd.notnull(results.loc["error", problem, config]) and \
                  results.loc["status", problem, config] == "optimal":
                    num_not_null += 1
                    total_error += results.loc["error", problem, config]
            if num_not_null > 0:
                avg_errors.loc[config] = total_error / num_not_null
            else:
                avg_errors.loc[config] = None

        print("Average errors:")
        print(avg_errors)

    # Relative performance:
    # For each pair of configs:
    rel_performance = pd.DataFrame(index = results.minor_axis, columns = results.minor_axis)
    for standard in rel_performance.index:
        for compare in rel_performance.columns:
            # Compute average ratio (compare/standard) runtimes for problems that 
            # were solved by both configurations
            num_solved_by_both = 0
            total_compare = 0.0
            total_standard = 0.0
            for problem in results.major_axis:
                standard_time = results.loc["solve_time", problem, standard]
                compare_time = results.loc["solve_time", problem, compare]
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
    avg_num_iters = pd.Series(index = results.minor_axis)
    for config in results.minor_axis:
        total_num_iters = 0
        num_not_null = 0.0
        for problem in results.major_axis:
            if pd.notnull(results.loc["num_iters", problem, config]) and \
               results.loc["status", problem, config] == "optimal":
                num_not_null += 1
                total_num_iters += results.loc["num_iters", problem, config]
        if num_not_null > 0:
            avg_num_iters.loc[config] = total_num_iters / num_not_null
        else:
            avg_num_iters.loc[config] = None
    print("Average number of iterations")
    print(avg_num_iters)

    # Linear regression (scaling):
    # https://stackoverflow.com/questions/19991445/run-an-ols-regression-with-pandas-data-frame
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as sm

    print("solve_time vs. num_scalar_variables")
    print("-----------------------------------")

    for config in results.minor_axis:
        configData = results.minor_xs(config) # returns a DataFrame
        nvars = configData.loc[:, "num_scalar_variables"]
        solvetimes = configData.loc[:, "solve_time"]
        df = pd.DataFrame({"X" : nvars, "Y" : solvetimes})
        ls_reg = sm.ols(formula="X ~ Y", data=df).fit()
        print("config: " + config)
        print(ls_reg.params)

    return


if __name__ == '__main__':
    main(get_command_line_args())  




