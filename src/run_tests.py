import TestFramework as tf
import DataVisualization as dv
import time
import pandas as pd

framework = tf.TestFramework(problemDir = "problems", configDir = "configs")
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
dv.plot_performance_profile(results)






        




