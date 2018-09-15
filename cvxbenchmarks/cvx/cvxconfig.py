from cvxbenchmarks.cvx.base import Config
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
    def read(cls, configFile, format="yaml"):
        if format == "yaml" or format == "yml":
            return cls.read_YAML(configFile)
        else: # pragma: no cover
            raise TypeError("Invalid config file format: " + format)


    @classmethod
    def read_YAML(cls, configFile):
        """Alternative constructor for loading a configuration from a text file.
        Loads a YAML file from |configFile|

        Parameters
        ----------
        configFile : string
            File path to the configuration file.

        Returns
        -------
        A list of CVXConfig objects found in this configFile
        """
        configs = []
        with open(configFile, 'r') as f:

            data = yaml.load_all(f, Loader=yaml.Loader)
            for d in data:
                configID = d["configID"]
                del d["configID"]
                configs.append(cls(configID, d))
        return configs

    def __repr__(self):
        return repr(self.configID) + ": " + repr(self.solver_opts)

