import ProblemGenerator as pg
import os

templates = [
    "least_squares",
    "lasso",
    "control",
    "portfolio_opt",
    "lp"
]

for problemID in templates:
    paramFile = os.path.join("templates", problemID+"_params.txt")
    reader = pg.TemplateReader(problemID, paramFile)
    template, params = reader.read()
    writer = pg.ProblemWriter(template, params)
    writer.write(problemID, "problems")