import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plot a bunch of stuff:

# Just provide functions for now (no OOP)

def plot_performance_profile(results, rel_max = 10e10):
    """Type: step plot
    x-axis : tau
    y-axis : performance 
    Plots the performance profile described in 
    Dolan, More 2001. "Benchmarking optimization software with performance profiles"

    Parameters
    ----------
    results : pandas.panel
        The panel containing the results of the testing.
    rel_max : float
        The default value of performance for the case when
        a solver fails to solve a problem. We should have
        rel_max >= performance_i,j for all solvers i and all
        problems j.

    """
    for config in results.axes[2]:
        num_problems = len(results.axes[1])
        x = sorted(results.loc["performance", :, config]) 
        y = [len([val for val in x if val <= tau])/float(num_problems) for tau in x]

        # Extend the line all the way to the left. (DO THIS FIRST)
        y = [len([val for val in x if val <= 1.0])/float(num_problems)] + y
        x = [1.0] + x
        # Extend the line all the way to the right. (DO THIS SECOND)
        x += [rel_max] 
        y += [y[-1]]

        # Step plot stuff:
        # http://joelotz.github.io/step-functions-in-matplotlib.html
        plt.step(x, y, label = config, drawstyle = "steps-post")

    plt.legend(loc = 'lower right')
    plt.xlim(1, 100)

    # Labels and things
    plt.title("Performance profiles for solve time")
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$P(r_{p,s} \leq \tau : 1 \leq s \leq n_s)$')


def plot_scatter_by_config_log(results, x_field, y_field):
    """Generic scatter plot for results panel.
    
    Parameters
    ----------
    results : pandas.panel
        The panel containing the results of the testing.
    x_field : list or string
        A list of fields in results (or a single string) to
        be summed and log-plotted on the x axis.
    y_field : list or string
        A list of fields in results (or a single string) to
        be summed and log-plotted on the y axis.
    """
    # Color-code by configuration
    config_colors = ['b', 'c', 'y', 'm', 'r']
    serieslist = []
    for i, config in enumerate(results.axes[2]):
        if type(x_field) is list:
            x = np.log10(pd.DataFrame.sum(results.loc[x_field, :, config], axis = 1).values)
        else:
            x = np.log10(results.loc[x_field, :, config].values)
        if type(y_field) is list:
            y = np.log10(pd.DataFrame.sum(results.loc[y_field, :, config], axis = 1).values)
        else:
            y = np.log10(results.loc[y_field, :, config].values)
        series = plt.scatter(x, y, marker = 'o', color = config_colors[i])
        serieslist.append(series)

    plt.legend(tuple(serieslist),
               tuple(results.axes[2]),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.title(str(y_field) + " vs. " + str(x_field))
    plt.xlabel("log " + str(x_field))
    plt.ylabel("log " + str(y_field))


def plot_histograms_by_config(results):
    """For each configuration, plots a histogram of the error tolerances.

    Parameters
    ----------
    results : pandas.panel
        The panel containing the results of the testing.
    """
    for config in results.axes[2]:
        plt.figure()
        x = results.loc["error", :, config].values
        plt.hist(x, bins = 10, log = True)
        plt.title("Errors for " + config)
        plt.xlabel(r'$\log{\frac{|p - p_mosek|}{t_{abs} + |p_mosek|}}$')
        plt.draw()



