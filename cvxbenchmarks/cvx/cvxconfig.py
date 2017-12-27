from cvxbenchmarks.base import Config
import cvxbenchmarks.settings as s

from ruamel import yaml

class CVXConfig(Config):
    """An object for managing the configuration of a cvx solver.

    Configuration files should be provided in YAML format (http://yaml.org).
    For example, the following data file (consisting of three lines)

    solver: SCS
    verbose: true
    eps: 1e-4

    ...should produce the following dictionary:

    {
        'solver': 'SCS',
        'eps': 0.0001,
        'verbose': True
    }

    Attributes
    ----------
    solver_opts : dict
        Dictionary of key-value pairs that constitute the parameters of the solver.
        Example:

    """

    def __init__(self, solver_opts=None):
        if solver_opts is None:
            self.solver_opts = {}
        else:
            self.solver_opts = solver_opts

    def configure(self):
        return self.solver_opts

    @classmethod
    def from_file(cls, configFile):
        """Alternative constructor for loading a configuration from a text file.
        Loads a YAML file from |configFile|

        Parameters
        ----------
        configFile : string
            File path to the configuration file.
        """
        with open(configFile, 'r') as f:
            return cls(yaml.safe_load(f))

