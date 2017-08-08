# analyze_scs_superscs

import pandas as pd
import numpy as np
import cvxbenchmarks.data_visualization as dv
import matplotlib.pyplot as plt

import math

print("Loading results from {}...".format("results.pkl"))
results = pd.read_pickle("results.pkl")
# results = pd.read_hdf(args.results)
print(results.to_frame(filter_observations = False))
print("Done.")

# Split into optimal and inaccurate:
scs_results = results.minor_xs("scs_config")
superscs_results = results.minor_xs("superscs_config")

scs_results_optimal = scs_results.loc[scs_results["status"] == "optimal"] 
superscs_results_optimal = superscs_results.loc[superscs_results["status"] == "optimal"] 
scs_results_inacc = scs_results.loc[scs_results["status"] == "optimal_inaccurate"] 
superscs_results_inacc = superscs_results.loc[superscs_results["status"] == "optimal_inaccurate"] 

# Filter out the problems that mosek did not solve.
mosek_results = results.minor_xs("mosek_config")
mosek_not_null = mosek_results.loc[pd.notnull(mosek_results["status"])]
# print(mosek_not_null)
# print(scs_results_optimal.index)
# print(superscs_results_optimal.index)
# print(scs_results_inacc.index)
# print(superscs_results_inacc.index)


# Ignore outlier portfolio_0
scs_opt_superscs_opt = scs_results_optimal.index & superscs_results_optimal.index #& mosek_not_null.index
scs_opt_superscs_opt = scs_opt_superscs_opt.delete(scs_opt_superscs_opt.get_loc("portfolio_0"))
print(scs_opt_superscs_opt)
scs_inacc_superscs_inacc = scs_results_inacc.index & superscs_results_inacc.index #& mosek_not_null.index
scs_inacc_superscs_opt = scs_results_inacc.index & superscs_results_optimal.index #& mosek_not_null.index

results_opt = results.loc[:, scs_opt_superscs_opt,:]
results_inacc = results.loc[:, scs_inacc_superscs_inacc,:]
results_mixed = results.loc[:, scs_inacc_superscs_opt,:]






for i, results in enumerate([results_opt, results_inacc, results_mixed]):
    # Latex:
    if i == 0:
        with open("opt_time_table.tex", "w") as f:
            f.write(results.loc["solve_time", :, ["superscs_config", "scs_config"]].to_latex())


    # Plots
    # Performance Profiles
    plt.figure()
    dv.plot_performance_profile(results, xmax=30)
    # dv.plot_performance_profile(results.loc[:,:,["superscs_config", "scs_config"]], xmax=30)
    plt.draw()

    # Graph time vs. number of scalar variables
    plt.figure()
    dv.plot_scatter_by_config(results, "num_scalar_variables", "solve_time", logx = True, logy = True)
    plt.draw()

    # Graph num_iterations vs. number of scalar variables
    # Graph time vs. number of scalar variables
    plt.figure()
    dv.plot_scatter_by_config(results, "num_scalar_variables", "num_iters", logx = True, logy = False)
    plt.draw()


    # Aggregate statistics

    # Errors
    avg_errors = pd.Series(index = results.minor_axis)
    for config in results.minor_axis:
        total_error = 0.0
        num_not_null = 0
        for problem in results.major_axis:
            if pd.notnull(results.loc["error", problem, config]):
                num_not_null += 1
                total_error += results.loc["error", problem, config]
        if num_not_null > 0:
            avg_errors.loc[config] = total_error / num_not_null
        else:
            avg_errors.loc[config] = None
        # print(total_error)
        # print(num_not_null)


    print("Average errors:")
    print(avg_errors)

    # Number of iterations
    avg_num_iters = pd.Series(index = results.minor_axis)
    for config in results.minor_axis:
        total_num_iters = 0
        num_not_null = 0.0
        for problem in results.major_axis:
            if pd.notnull(results.loc["num_iters", problem, config]):
                num_not_null += 1
                total_num_iters += results.loc["num_iters", problem, config]
        if num_not_null > 0:
            avg_num_iters.loc[config] = total_num_iters / num_not_null
        else:
            avg_num_iters.loc[config] = None
    print("Average number of iterations")
    print(avg_num_iters)

    # Solve Time
    avg_solve_time = pd.Series(index = results.minor_axis)
    for config in results.minor_axis:
        total_time = 0.0
        num_not_null = 0
        for problem in results.major_axis:
            if pd.notnull(results.loc["solve_time", problem, config]):
                num_not_null += 1
                total_time += results.loc["solve_time", problem, config]
        if num_not_null > 0:
            avg_solve_time.loc[config] = total_time / num_not_null
        else:
            avg_solve_time.loc[config] = None
    print("Average solve time:")
    print(avg_solve_time)

    # Percentage of problems solved faster
    scs_better = 0
    superscs_better = 0
    total = 0
    for problem in results.major_axis:
            if pd.notnull(results.loc["solve_time", problem, config]):
                if results.loc["solve_time", problem, "scs_config"] > \
                   results.loc["solve_time", problem, "superscs_config"]:
                   superscs_better += 1
                   total += 1
                elif results.loc["solve_time", problem, "scs_config"] < \
                   results.loc["solve_time", problem, "superscs_config"]:
                   scs_better += 1
                   total += 1
                else:
                    print("tie!")
                    total+= 1
    print("Percentage of better times:")
    print("\tsuperscs better: {}".format(superscs_better*100.0/total))
    print("\tscs better: {}".format(scs_better*100.0/total))






for i in plt.get_fignums():
    plt.figure(i)
    plt.tight_layout()
    plt.savefig('figure{}.png'.format(i))
