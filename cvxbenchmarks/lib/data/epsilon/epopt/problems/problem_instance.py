
from collections import namedtuple

class ProblemInstance(namedtuple("ProblemInstance", ["name", "f", "kwargs"])):
    def create(self):
        t = self.f(**self.kwargs)
        if isinstance(t, tuple):
            setattr(t[0], 'kwargs', self.kwargs)
        else:
            setattr(t, 'kwargs', self.kwargs)
        return t
