# render_template.py
# Tests the rendering functionality for specific jinja2 templates
# Renders only the first line of the parameter file

import cvxbenchmarks.problem_generator as pg
import sys
import os.path

from cvxbenchmarks.scripts.utils import get_command_line_args

#templateDir = os.path.join("lib", "data", "cvxbenchmarks")


def main(args):
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

if __name__ == '__main__':
    main(get_command_line_args())
