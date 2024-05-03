import numpy as np
from mpi4py import MPI

from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex_timeforcing
from pySDC.projects.GPU.problem_classes.rain import Rain
from mpi4py_fft import newDistArray



class allencahn_imex_timeforcing_adaptivity(allencahn_imex_timeforcing, Rain):
    r"""
    Add more source terms to `allencahn_imex_timeforcing` such that the time-scale changes and we can benefit from adaptivity.
    """
    t_next_drop = 0
    time_between_drops = 1e-3

    def __init__(self, time_freq=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._makeAttributeAndRegister('time_freq', localVars=locals(), readOnly=True)
        self.set_random_generator(0)

        self.get_random_params()

    def eval_f2(self, u, t):
        if t > self.t_next_drop:
            self.get_random_params()

            self.rain_params = u.comm.bcast(self.rain_params)
            self.logger.log(15, f'New rain drop with coordinates {self.rain_params}')
            self.t_next_drop += self.time_between_drops

        you = self.single_drop()

        me = super().eval_f(u, t)
        me.expl += you
        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        assert not self.spectral

        f = self.dtype_f(self.init)

        f.impl[:] = self._eval_Laplacian(u, f.impl)

        if self.eps > 0:
            f.expl = -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u)

        # build sum over RHS without driving force
        Rt_local = float(self.xp.sum(f.impl + f.expl))
        if self.comm is not None:
            Rt_global = self.comm.allreduce(sendobj=Rt_local, op=MPI.SUM)
        else:
            Rt_global = Rt_local

        # build sum over driving force term
        Ht_local = float(self.xp.sum(6.0 * u * (1.0 - u)))
        if self.comm is not None:
            Ht_global = self.comm.allreduce(sendobj=Ht_local, op=MPI.SUM)
        else:
            Ht_global = Rt_local

        # add/substract time-dependent driving force
        if Ht_global != 0.0:
            dw = Rt_global / Ht_global
        else:
            dw = 0.0

        time_dep_fac = np.cos(self.time_freq * 2 * np.pi * t)**2
        f.expl -= 6.0 * dw * u * (1.0 - u * time_dep_fac)

        self.work_counters['rhs']()
        return f
