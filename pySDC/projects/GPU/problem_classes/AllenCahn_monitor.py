import numpy as np

from pySDC.projects.TOMS.AllenCahn_monitor import monitor


class monitor_MPI(monitor):
    phase_thresh = 0.5  # count everything above this threshold to the high phase.

    @classmethod
    def get_radius(cls, u, L):
        cls.xp = L.prob.xp
        dx = float(L.prob.dx)
        comm = L.prob.comm
        c = float(comm.allreduce(cls.xp.count_nonzero(u > cls.phase_thresh)))
        return np.sqrt(c / np.pi) * dx

    @classmethod
    def get_interface_width(cls, u, L):
        return float(super().get_interface_width(u, L))
