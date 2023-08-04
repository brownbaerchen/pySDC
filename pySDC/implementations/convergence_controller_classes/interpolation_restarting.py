import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController, Pars
from pySDC.core.Lagrange import LagrangeApproximation


class InterpolationRestarting(ConvergenceController):
    """ """

    @classmethod
    def get_implementation(cls, useMPI):
        """
        Retrieve the implementation for a specific flavor of this class.

        Args:
            useMPI (bool): Whether or not the controller uses MPI

        Returns:
            cls: The child class that implements the desired flavor
        """
        if useMPI:
            raise NotImplementedError
        else:
            return InterpolationRestartingNonMPI

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": 150,
            "gamma": 1,
        }

        return {**defaults, **super().setup(controller, params, description, **kwargs)}


class InterpolationRestartingNonMPI(InterpolationRestarting):
    def get_uend(self, controller, uend, time, MS, active_slots, **kwargs):
        """
        Replace restart with interpolation to the maximal allowed step size.

        Args:
            controller (pySDC.Controller): The controller
            uend (dtype_u): Initial conditions for next step
            time (list): Times of the next step
            MS (list): List of steps
            active_slots (list): Indices of active steps

        Returns:
            dtype_u: Initial conditions for the next block
            list: Initial time for the next block
        """
        restarts = [MS[i].status.restart for i in range(len(MS)) if i in active_slots]
        restart_at = np.where(restarts)[0][0] if True in restarts else len(restarts)

        if True in restarts:
            lvl = MS[restart_at].levels[0]

            if not lvl.status.dt_new:
                return uend, time

            u_old = [me.flatten() for me in lvl.u]

            # interpolate
            new_time = lvl.time + lvl.status.dt_new * self.params.gamma
            interpolator = LagrangeApproximation(points=lvl.time + np.append([0], lvl.sweep.coll.nodes) * lvl.params.dt)

            uend[:] = (interpolator.getInterpolationMatrix([new_time]) @ u_old)[0].reshape(lvl.u[0].shape)
            time[active_slots[0]] = new_time

            MS[restart_at].status.restarts_in_a_row = 0
            self.logger.info(
                f'Overwriting restart: Starting next block with solution interpolated to t={time[active_slots[0]]:.2f}'
            )

        return uend, time
