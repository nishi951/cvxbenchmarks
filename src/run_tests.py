import TestFramework as tf
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

LARGE_VALUE = 10e10

performance = pd.DataFrame(index = results.axes[1], columns = results.axes[2])
num_problems = 0
for problem in results.axes[1]:
    num_problems += 1
    best = LARGE_VALUE
    for config in results.axes[2]:
        # Get best performance for each problem.
        this = results.loc["time", problem, config]
        if this < best: # also works if this is NaN
            best = this

    if best == LARGE_VALUE:
        # No solver could solve this problem.
        print "all solvers failed on",problem
        for config in results.axes[2]:
            performance.loc[problem, config] = LARGE_VALUE;
        continue


    else: # Compute t/t_best for each problem for each config
        for config in results.axes[2]:
            if math.isnan(results.loc["time", problem, config]):
                performance.loc[problem, config] = LARGE_VALUE
            else:
                performance.loc[problem, config] = results.loc["time", problem, config]/best

results["performance"] = performance

# Graph performance profile:
for config in results.axes[2]:
    x = sorted(results.loc["performance", :, config]) 
    y = [len([val for val in x if val < tau])/float(num_problems) for tau in x]
    
    # Extend the line all the way to the right.
    x += [LARGE_VALUE] 
    y += [y[-1]]
    plt.step(x, y, label = config)

plt.legend(loc = 'lower right')
plt.xlim(1, 10)

# Labels and things
plt.title("Performance profiles for solve time")
plt.xlabel(r'$\tau$')
plt.ylabel(r'$P(r_{p,s} \leq \tau : 1 \leq s \leq n_s)$')

plt.show()






        




