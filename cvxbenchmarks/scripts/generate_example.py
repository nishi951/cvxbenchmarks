import ProblemGenerator as pg
import os

template_files = [
    ("least_squares.j2", "least_squares_params.csv"),
    ("lasso.j2", "lasso_params.csv"),
    ("control.j2", "control_params.csv"),
    ("portfolio_opt.j2", "portfolio_opt_params.csv"),
    ("lp.j2", "lp_params.csv"),
    ("huber.j2", "huber_params.csv")
]

templateDir = os.path.join("lib", "data", "cvxbenchmarks")

# Generate problems from templates
for templateFile, paramFile in template_files:
    paramFile = os.path.join(templateDir, paramFile)
    templ = pg.ProblemTemplate(templateFile, paramFile, templateDir)
    templ.write_to_dir("problems")

# Generate index file
index = pg.Index("problems")
index.write()

