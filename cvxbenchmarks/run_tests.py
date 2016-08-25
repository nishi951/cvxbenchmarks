import TestFramework as tf
import DataVisualization as dv
import time
import pandas as pd

framework = tf.TestFramework(problemDir = "problems", configDir = "lib/configs")
print "Loading problems..."
framework.preload_all_problems()
print "\tDone."

print "Loading configs..."
framework.preload_all_configs()
print "\tDone."

start = time.time()
# print "Solving all problem instances..."
# framework.solve_all()
# print "\tDone."


print "Solving all problem instances in parallel..."
framework.solve_all_parallel()
print "\tDone."
print "\tTime:",str(time.time()-start)

print "number of results:", str(len(framework.results))

# Export results to a pandas panel
print "exporting results."
results = framework.export_results_as_panel()
print results.to_frame(filter_observations = False)

# Data Visualization
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

# Graph histogram of solve accuracies (relative to mosek)
# dv.plot_histograms_by_config(results)

plt.show()







        




