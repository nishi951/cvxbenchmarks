import TestFramework as tf
import time
import pandas as pd

# NOTE: Do not import matplotlib.pyplot before running TestFramework's solve_all_parallel - 
# Something about matplotlib.pyplot messes up the solve (at least in a virtualenv)

framework = tf.TestFramework(problemDir = "problems", configDir = "lib/configs")
print("Loading problems...")
framework.preload_all_problems()
# framework.load_problem_file("least_squares_0")
print("\tDone.")

print("Loading configs...")
# framework.preload_all_configs()
# framework.load_config("ecos_config")
framework.load_config("scs_config")
framework.load_config("superscs_config")
framework.load_config("mosek_config")
print("\tDone.")

start = time.time()
print("Solving all problem instances...")
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
results.to_hdf("results.dat", key = "results", mode = "w")

# Data Visualization
import DataVisualization as dv
import matplotlib.pyplot as plt
import math

# Generate performance profiles for all solver configurations

# Graph performance profile:
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
dv.plot_scatter_by_config(results, "num_scalar_variables", "num_iterations", logx = True, logy = False)
plt.draw()

# Graph histogram of solve accuracies (relative to mosek)
# dv.plot_histograms_by_config(results)

# Show figures
# plt.show()

# Save figures to a single file:
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







        




