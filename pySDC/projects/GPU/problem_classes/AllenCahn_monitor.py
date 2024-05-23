import numpy as np

from pySDC.projects.TOMS.AllenCahn_monitor import monitor


class monitor_MPI(monitor):
    phase_thresh = 0.5  # count everything above this threshold to the high phase.

    @classmethod
    def get_radius(cls, u, L):
        cls.xp = L.prob.xp
        dx = L.prob.dx
        comm = L.prob.comm
        c = float(comm.allreduce(np.count_nonzero(u > cls.phase_thresh)))
        return np.sqrt(c / np.pi) * dx
