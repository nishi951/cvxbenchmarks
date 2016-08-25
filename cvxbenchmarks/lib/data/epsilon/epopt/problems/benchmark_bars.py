
import matplotlib.pyplot as plt
import numpy as np

import sys

COLORS = ["b", "r", "g"]
BENCHMARKS = ["epsilon", "scs", "ecos"]
WIDTH = 0.2

if __name__ == "__main__":
    results = {}
    for line in sys.stdin:
        benchmark, problem, time, value = line.split()
        results[(problem, benchmark)] = (float(time), float(value))

    problems = set(k[0] for k in results)
    problems = list(problems)
    problems.sort()

    x = np.arange(len(problems))
    ax = plt.subplot(111)

    # for i, benchmark in enumerate(BENCHMARKS):
    #     print [results.get((p, benchmark), (0,0))[0] for p in problems]

    for i, benchmark in enumerate(BENCHMARKS):
        ax.bar(x+i*WIDTH,
               [results.get((p, benchmark), (0,0))[0] for p in problems],
               log=True, width=WIDTH, color=COLORS[i])

    plt.autoscale(tight=True)

    plt.ylabel("Running time (seconds)")
    plt.ylim((5e-2, 1e4))

    plt.xticks(x+0.3, problems)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)

    plt.legend(("Epsilon", "CVXPY+SCS", "CVXPY+ECOS"), loc="best", ncol=3)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(sys.stdout, format="pdf")
