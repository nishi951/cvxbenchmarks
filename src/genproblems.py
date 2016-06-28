from jinja2 import FileSystemLoader, Environment

# Generate Problems

# - Least squares
def make_problem(template, kwargs):

    return 0

# - Control

# - Portfolio Optimization
# - Signal Processing
#      - Filter design
# - LP
# - LASSO
# - Regression
# - Classifiers
# - Network problems
# - Dynamic Energy
# - Radiation Treatment planning
# - Misc.
#      - Pathological

if __name__ == "__main__":
    # The below maybe should go in a problem configuration file that gets
    # imported to this script
    problemTypes = ["least_squares", 
                    "control", 
                    "portfolio_opt", 
                    # "signal_processing", 
                    "lp", 
                    "lasso"
                    # "regression",
                    # "classifier",
                    # "network",
                    # "dynamic_energy",
                    # "radiation_treatment",
                    # "misc"
    ]

    sizes = {
        "small": 100,
        "medium": 10000,
        "large": 1000000
    }

    env = Environment(loader = FileSystemLoader("templates"))
    for problemType in problemTypes:
        for size, base in sizes.items(): # small, medium, large
            for seed in range(1,6): # set 5 different random seeds
                # Generate nproblems problems for each size category for each problem
                problemTemplate = env.get_template(problemType+".j2")
                # Alter the next line to make platform-independent?
                with open("problems/"+problemType+"_"+size+"_"+str(seed)+".py", "wb") as f:
                    f.write(problemTemplate.render(seed = seed, base = base))


