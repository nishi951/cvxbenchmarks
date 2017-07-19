# render_template.py
# Tests the rendering functionality for specific jinja2 templates
# Renders only the first line of the parameter file

import ProblemGenerator as pg
import sys
import os.path


#templateDir = os.path.join("lib", "data", "cvxbenchmarks")
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("render_template.py: Usage: python render_template.py /path/to/template.j2 /path/to/params.csv")
    print("python render_template.py /path/to/template.j2")


templateFile = os.path.basename(sys.argv[1])
templateDir = os.path.dirname(sys.argv[1])
paramFile = sys.argv[2]

# Generate problems from templates
if len(sys.argv) == 3:
    templ = pg.ProblemTemplate(templateFile, paramFile, templateDir)
else if len(sys.argv) == 2:
    templ = pg.ProblemTemplate(templateFile, None, templateDir)
print(str(templ))
