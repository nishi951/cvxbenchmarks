# Plot panel from pickle
import pandas as pd
import matplotlib.pyplot as plt
import DataVisualization as dv

results = pd.read_pickle("results.dat")

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

plt.show()
