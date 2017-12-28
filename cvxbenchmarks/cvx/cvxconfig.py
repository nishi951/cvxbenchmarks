from cvxbenchmarks.base import Config
import cvxbenchmarks.settings as s

from ruamel import yaml

class CVXConfig(Config):
    """An object for managing the configuration of a cvx solver.

    Configuration files should be provided in YAML format (http://yaml.org).
    For example, the following data file (consisting of four lines)

    configID: SCS_config
    solver: SCS
    verbose: true
    eps: 1e-4

    ...should produce the following dictionary:

    {
        'solver': 'SCS',
        'eps': 0.0001,
        'verbose': True
    }

    ...and set the configID of this CVXConfig to be "SCS_config".

    Important: configID and solver are MANDATORY, all others are optional.

    Attributes
    ----------
    configID : string
        A unique identifier for this configuration.
    solver_opts : dict
        Dictionary of key-value pairs that constitute the parameters of the solver.
        Example:

    """

    def __init__(self, configID, solver_opts=None):
        self.configID = configID
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
            data = yaml.safe_load(f)
            configID = data["configID"]
            del data["configID"]
            return cls(configID, data)

