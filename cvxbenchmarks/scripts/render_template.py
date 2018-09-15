# render_template.py
# Tests the rendering functionality for specific jinja2 templates
# Renders only the first line of the parameter file

import cvxbenchmarks.problem_generator as pg
import sys
import os.path

from cvxbenchmarks.scripts.utils import get_command_line_args

#templateDir = os.path.join("lib", "data", "cvxbenchmarks")


def main(args=None):
    templateFile, paramFile = args.template_and_params
    templ = pg.ProblemTemplate.from_file(templateFile, paramFile)
    print(str(templ))

if __name__ == '__main__':
    main(get_command_line_args())
