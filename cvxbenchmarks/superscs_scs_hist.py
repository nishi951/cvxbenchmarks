# Find histogram of superscs_times/scs_times
# Mark Nishimura 2017

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load results.dat
print("Loading results...")
# results = pd.read_hdf("results.dat")
results = pd.read_pickle("results.pkl")
print("Done.")

optimalratios = []
optimalinaccurateratios = []
for problem in results.major_axis:
    if results.loc["solve_time", problem, "superscs_config"] is not None and \
       results.loc["solve_time", problem, "scs_config"] is not None and \
       results.loc["solve_time", problem, "scs_config"] > 0:

       if results.loc["status", problem, "superscs_config"] == "optimal" and \
          results.loc["status", problem, "scs_config"] == "optimal":
            optimalratios += [results.loc["solve_time", problem, "superscs_config"]/results.loc["solve_time", problem, "scs_config"]]

       elif results.loc["status", problem, "superscs_config"] == "optimal_inaccurate" and \
          results.loc["status", problem, "scs_config"] == "optimal_inaccurate":
            optimalratios += [results.loc["solve_time", problem, "superscs_config"]/results.loc["solve_time", problem, "scs_config"]]

optimalratios = [ratio for ratio in optimalratios if not np.isnan(ratio)]
optimalinaccurateratios = [ratio for ratio in optimalinaccurateratios if not np.isnan(ratio)]

print(len(optimalratios))
plt.figure()
plt.hist(x = optimalratios, bins = np.logspace(-1, 2, 20), log = False)
plt.xscale('log')
plt.title("histogram of superscs time / scs time (status = optimal)")
plt.xlabel("superscs time / scs time")
plt.draw()

print(len(optimalinaccurateratios))
plt.figure()
plt.hist(x = optimalinaccurateratios, bins = np.logspace(-1, 5, 20), log = False)
plt.xscale('log')
plt.title("histogram of superscs time / scs time (status = optimal_inaccurate")
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
