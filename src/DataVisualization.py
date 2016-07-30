import pandas as pd
import matplotlib.pyplot as plt
import math

# Load results from file?
# Plot a bunch of stuff:

# Just provide functions for now (no OOP)

def plot_performance_profile(results, rel_max = 10e10):
    """Plots the performance profile described in 
    Dolan, More 2001. "Benchmarking optimization software with performance profiles"
    """
    for config in results.axes[2]:
        num_problems = len(results.axes[1])
        x = sorted(results.loc["performance", :, config]) 
        y = [len([val for val in x if val < tau])/float(num_problems) for tau in x]
        
        # Extend the line all the way to the right.
        x += [rel_max] 
        y += [y[-1]]
        plt.step(x, y, label = config)

    plt.legend(loc = 'lower right')
    plt.xlim(1, 100)

    # Labels and things
    plt.title("Performance profiles for solve time")
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$P(r_{p,s} \leq \tau : 1 \leq s \leq n_s)$')

    plt.show()

