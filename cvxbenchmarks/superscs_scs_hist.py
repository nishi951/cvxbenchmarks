# Find histogram of superscs_times/scs_times
# Mark Nishimura 2017

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load results.dat
print("Loading results...")
results = pd.read_hdf("results.dat")
print("Done.")

ratios = []
for problem in results.major_axis:
    if results.loc["solve_time", problem, "superscs_config"] is not None and \
       results.loc["solve_time", problem, "scs_config"] is not None and \
       results.loc["solve_time", problem, "scs_config"] > 0:
        ratios += [results.loc["solve_time", problem, "superscs_config"]/results.loc["solve_time", problem, "scs_config"]]
        ratios = [ratio for ratio in ratios if not np.isnan(ratio)]
plt.figure()
plt.hist(x =ratios, bins = np.logspace(0, 4, 20), log = False)
plt.xscale('log')
plt.title("histogram of superscs time / scs time")
plt.xlabel("superscs time / scs time")
plt.draw()


# Save figures to a single file:
# http://stackoverflow.com/questions/26368876/saving-all-open-matplotlib-figures-in-one-file-at-once
from matplotlib.backends.backend_pdf import PdfPages

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

multipage("superscs_scs_hist.pdf")
