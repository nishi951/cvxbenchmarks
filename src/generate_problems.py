import ProblemGenerator as pg
import os

templates = [
    "least_squares",
    "lasso",
    "control",
    "portfolio_opt",
    "lp"
]

# Generate problems from templates
for problemID in templates:
    paramFile = os.path.join("templates", problemID+"_params.txt")
    templ = pg.ProblemTemplate(problemID, paramFile)
    templ.write("problems")

# Generate index file
index = pg.Index("problems")
index.write()
