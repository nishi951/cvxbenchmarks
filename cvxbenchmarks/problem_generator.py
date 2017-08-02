from jinja2 import Environment, FileSystemLoader
import re, os, sys, inspect
import numpy as np
import pandas as pd

# cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
# if cvxfolder not in sys.path:
    # sys.path.insert(0, cvxfolder) 
# sys.path.insert(0, "/Users/mark/Documents/Stanford/reu2016/cvxpy")
import cvxpy as cvx
print(cvx)

# Format of the Parameter file:
# Csv file readable by pandas
# ex.
# problemID, m, n, seed
# ls_0, 20, 30, 1
# ls_1, 40, 60, 1
# ls_big 500, 600, 1


class ProblemTemplate(object):
    """A templated version of a cvxpy problem that can be read and written.

    Stores one jinja2 template and potentially multiple sets of 
    parameters for it.

    Attributes
    ----------
    template : jinja2.template
        A jinja2 template containing the problem we wish to render.
    params : list of dict
        A list of dictionary objects, one for each instance of the template 
        we want to render.
    """
    # https://pythonconquerstheuniverse.wordpress.com/2012/02/15/mutable-default-arguments/
    def __init__(self, name="defaultTemplate", template=None, params=None):
        self.name = name
        self.template = template
        if params is None:
            self.params = []
        else:
            self.params = params

    @classmethod
    def from_file(cls, templateFile, paramFile="", name="defaultTemplate"):
        """Alternative constructor for loading a ProblemTemplate 
        directly from files.

        """
        # print("from_file: {}".format(params))
        newTemplate = cls(name=name)
        newTemplate.read(templateFile, paramFile)
        return newTemplate

    def read_template(self, templateFile):
        """Read in a template. Uses os.path.dirname and os.path.basename
        to retrieve the template name and the environment directory 
        from the templateFile argument.

        Parameters
        ----------
        templateFile : string
            The filepath to the (.j2) file containing the problem template.

        Returns
        -------
        template : jinja2.Template
            The template stored in the file in templateFile.
        """
        templateDir = os.path.dirname(templateFile)
        templateName = os.path.basename(templateFile)
        self.name = templateName
        try:
            env = Environment(loader=FileSystemLoader(templateDir))
            self.template = env.get_template(templateName)
        except Exception as e:
            print(("Problem loading template {template} "
                    "in {templateDir}. "
                  ).format(template=templateName, templateDir=templateDir))
            print(e)
            self.template = None
        return

    def read_param_csv(self, paramFile, overwrite):
        """Read in a parameter file csv and store each row as a dictionary.

        Parameters
        ----------
        paramFile : string
            The filepath to the (.csv) file containing the parameters
            in csv format. 
            e.g.
            ----
            problemID, m, n, seed
            basis_pursuit_0, 100, 300, 0
            basis_pursuit_1, 200, 400, 0
            basis_pursuit_2, 300, 500, 0
            ----

        Returns
        -------
        params : list of dict
            The list contains one dictionary per instance of template 
            to generate.
            e.g. The above example would generate
            [
                {
                    "problemID": "basis_pursuit_0",
                    "m": 100,
                    "n": 300,
                    "seed": 0
                },
                ...
            ]
        """
        params = []
        try:
            paramsDf = pd.read_csv(paramFile, skipinitialspace=True)
            for _, row in paramsDf.iterrows():
                params.append(row.to_dict())
        except Exception as e:
            print(("Problem loading parameters in {}. "
                   "Check parameter file path."
                   ).format(paramFile))
        if overwrite:
            self.params = params
        else:
            self.params.extend(params)
        return

    def read(self, templateFile, paramFile, overwrite = False):
        """Read in the template and the parameters.
        """
        self.read_template(templateFile)
        if paramFile is not None:
            self.read_param_csv(paramFile, overwrite)
        return

    def write_to_dir(self, problemDir):
        """Write all the template instances to files.

        Parameters
        ----------
        problemDir : string
            The directory where the template instances should be written.
        """
        try:
            for paramDict in self.params:
                instanceID = paramDict["problemID"]
                print(open)
                with open(os.path.join(problemDir, 
                                       instanceID + ".py"), "w") as f:
                    f.write(self.template.render(paramDict))
        except Exception as e:
            print("Unable to render template: {}".format(self.name))
            print(e)

    def __str__(self):
        if len(self.params) > 0:
            return self.template.render(self.params[0])
        return self.template.render()




