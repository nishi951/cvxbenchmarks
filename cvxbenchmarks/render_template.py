# render_template.py
# Tests the rendering functionality for specific jinja2 templates
# Renders only the first line of the parameter file

import ProblemGenerator as pg
import sys
import os.path


#templateDir = os.path.join("lib", "data", "cvxbenchmarks")


# Generate problems from templates
if len(sys.argv) == 3:
    templateFile = sys.argv[1]
    paramFile = sys.argv[2]
    templ = pg.ProblemTemplate.from_file(templateFile, paramFile)
    print(str(templ))


elif len(sys.argv) == 2:
    templateFile = sys.argv[1]
    templ = pg.ProblemTemplate.from_file(templateFile, None)
    print(str(templ))

else:
    print("render_template.py: Usage: python render_template.py /path/to/template.j2 /path/to/params.csv")
    print("python render_template.py /path/to/template.j2")
