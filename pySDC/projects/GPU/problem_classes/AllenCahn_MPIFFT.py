import numpy as np
from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex_timeforcing



class allencahn_imex_timeforcing_adaptivity(allencahn_imex_timeforcing):
    r"""
    Add more source terms to `allencahn_imex_timeforcing` such that the time-scale changes and we can benefit from adaptivity.
    """

    def __init__(self, time_freq=1., time_dep_strength=3., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._makeAttributeAndRegister('time_freq', 'time_dep_strength', localVars=locals(), readOnly=True)

    def eval_f(self, u, t):
        f = super().eval_f(u, t)
        f.expl *= self.get_time_dep_fac(self.time_freq, self.time_dep_strength, t)
        return f

    @staticmethod
    def get_time_dep_fac(time_freq, time_dep_strength, t):
        return time_dep_strength *  np.cos(time_freq * 2 * np.pi * t)**2 + 1# - time_dep_strength
