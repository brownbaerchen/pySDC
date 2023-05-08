import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld


class CheckIterationEstimator(ConvergenceController):
    def __init__(self, controller, params, description, **kwargs):
        """
        Initalization routine

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        super().__init__(controller, params, description)
        self.status = Status(["diff_old_loc", "diff_first_loc"])

    @classmethod
    def get_implementation(cls, useMPI, **kwargs):
        """
        Get MPI or non-MPI version

        Args:
            useMPI (bool): The implementation that you want

        Returns:
            cls: The child class implementing the desired flavor
        """
        if useMPI:
            return CheckIterationEstimatorMPI
        else:
            return CheckIterationEstimatorNonMPI

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        In this case, we need the user to supply a tolerance.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if "errtol" not in params.keys():
            return (
                False,
                "Please give the iteration estimator a tolerance in the form of `errtol`. Thanks!",
            )

        return True, ""

    def setup(self, controller, params, description, **kwargs):
        """
        Setup parameters. Here we only give a default value for the control order.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: The updated parameters
        """
        return {"control_order": -50, **super().setup(controller, params, description, **kwargs)}

    def dependencies(self, controller, description, **kwargs):
        """
        Need to store the solution of previous iterations.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        controller.add_convergence_controller(StoreUOld, description=description)
        return None


class CheckIterationEstimatorNonMPI(CheckIterationEstimator):
    def __init__(self, controller, params, description, **kwargs):
        """
        Initalization routine

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        super().__init__(controller, params, description)
        self.buffers = Status(["Kest_loc", "diff_new", "Ltilde_loc"])

    def setup_status_variables(self, controller, MS, **kwargs):
        """
        Setup storage variables for the differences between sweeps for all steps.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        self.status.diff_old_loc = [0.0] * len(MS)
        self.status.diff_first_loc = [0.0] * len(MS)
        return None

    def reset_buffers_nonMPI(self, controller, **kwargs):
        """
        Reset buffers used to imitate communication in non MPI version.

        Args:
            controller (pySDC.controller): The controller

        Returns:
            None
        """
        self.buffers.Kest_loc = [99] * len(controller.MS)
        self.buffers.diff_new = 0.0
        self.buffers.Ltilde_loc = 0.0

    def check_iteration_status(self, controller, S, **kwargs):
        """
        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.step): The current step

        Returns:
            None
        """
        L = S.levels[0]
        slot = S.status.slot

        # find the global maximum difference between iterations
        for m in range(1, L.sweep.coll.num_nodes + 1):
            self.buffers.diff_new = max(self.buffers.diff_new, abs(L.uold[m] - L.u[m]))

        if S.status.iter == 1:
            self.status.diff_old_loc[slot] = self.buffers.diff_new
            self.status.diff_first_loc[slot] = self.buffers.diff_new
        elif S.status.iter > 1:
            # approximate contraction factor
            self.buffers.Ltilde_loc = min(self.buffers.diff_new / self.status.diff_old_loc[slot], 0.9)

            self.status.diff_old_loc[slot] = self.buffers.diff_new

            # estimate how many more iterations we need for this step to converge to the desired tolerance
            alpha = 1 / (1 - self.buffers.Ltilde_loc) * self.status.diff_first_loc[slot]
            self.buffers.Kest_loc = np.log(self.params.errtol / alpha) / np.log(self.buffers.Ltilde_loc) * 1.05
            self.logger.debug(
                f'LOCAL: {L.time:8.4f}, {S.status.iter}: {int(np.ceil(self.buffers.Kest_loc))}, '
                f'{self.buffers.Ltilde_loc:8.6e}, {self.buffers.Kest_loc:8.6e}, \
{self.buffers.Ltilde_loc ** S.status.iter * alpha:8.6e}'
            )

            # set global Kest as last local one, force stop if done
            if S.status.last:
                Kest_glob = self.buffers.Kest_loc
                if np.ceil(Kest_glob) <= S.status.iter:
                    S.status.force_done = True


class CheckIterationEstimatorMPI(CheckIterationEstimator):
    from mpi4py import MPI

    INT = MPI.INT
    DOUBLE = MPI.DOUBLE

    def setup_status_variables(self, controller, **kwargs):
        """
        Setup storage variables for the differences between sweeps for all steps.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        self.status.diff_old_loc = 0.0
        self.status.diff_first_loc = 0.0
        return None

    def check_iteration_status(self, controller, S, comm, **kwargs):
        """
        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.step): The current step
            comm (mpi4py.MPI.Intracomm): Communicator

        Returns:
            None
        """
        # print(f'{comm.rank} is waiting before Barrier! For {comm.size-1} others. Prev: {S.prev}, next: {S.next}, {S.status.prev_done}', flush=True)
        # comm.Barrier()
        # print(f'{comm.rank} is past Barrier!', flush=True)
        L = S.levels[0]
        slot = S.status.slot
        diff_new = 0.0

        # find the global maximum difference between iterations
        for m in range(1, L.sweep.coll.num_nodes + 1):
            diff_new = max(diff_new, abs(L.uold[m] - L.u[m]))

        # send forward diff
        for hook in controller.hooks:
            hook.pre_comm(step=S, level_number=0)

        print(f'{comm.rank} is at 1!', flush=True)

        if S.status.force_done:
            return None

        if not (S.status.first or S.status.prev_done):
            # prev_diff = np.empty(1, dtype=float)
            # comm.Recv((prev_diff, CheckIterationEstimatorMPI.DOUBLE), source=S.prev, tag=999)
            prev_diff = comm.recv(source=S.prev, tag=999)
            if S.status.force_done:
                return None
            controller.logger.debug(
                'recv diff: status %s, process %s, time %s, source %s, tag %s, iter %s'
                % (prev_diff, S.status.slot, S.time, S.prev, 999, S.status.iter)
            )
            diff_new = max(prev_diff, diff_new)

        if not (S.status.last or S.status.done):
            controller.logger.debug(
                'send diff: status %s, process %s, time %s, target %s, tag %s, iter %s'
                % (diff_new, S.status.slot, S.time, S.next, 999, S.status.iter)
            )
            # tmp = np.array(diff_new, dtype=float)
            # comm.Send((tmp, CheckIterationEstimatorMPI.DOUBLE), dest=S.next, tag=999)
            comm.send(diff_new, dest=S.next, tag=999)

        for hook in controller.hooks:
            hook.post_comm(step=S, level_number=0)
        print(f'{comm.rank} is at 2!', flush=True)

        # Store values from first iteration
        if S.status.iter == 1:
            self.status.diff_old_loc = diff_new
            self.status.diff_first_loc = diff_new
        # Compute iteration estimate
        elif S.status.iter > 1:
            Ltilde_loc = min(diff_new / self.status.diff_old_loc, 0.9)
            self.status.diff_old_loc = diff_new
            alpha = 1 / (1 - Ltilde_loc) * self.status.diff_first_loc
            Kest_loc = np.log(S.params.errtol / alpha) / np.log(Ltilde_loc) * 1.05  # Safety factor!
            controller.logger.debug(
                f'LOCAL: {L.time:8.4f}, {S.status.iter}: {int(np.ceil(Kest_loc))}, '
                f'{Ltilde_loc:8.6e}, {Kest_loc:8.6e}, '
                f'{Ltilde_loc ** S.status.iter * alpha:8.6e}'
            )
            Kest_glob = Kest_loc
            # If condition is met, send interrupt
            if np.ceil(Kest_glob) <= S.status.iter:
                if S.status.last:
                    controller.logger.debug(f'{S.status.slot} is done, broadcasting..')
                    for hook in controller.hooks:
                        hook.pre_comm(step=S, level_number=0)
                    comm.Ibcast((np.array([1]), CheckIterationEstimatorMPI.INT), root=S.status.slot).Wait()
                    for hook in controller.hooks:
                        hook.post_comm(step=S, level_number=0, add_to_stats=True)
                    controller.logger.debug(f'{S.status.slot} is done, broadcasting done')
                    S.status.done = True
                else:
                    for hook in controller.hooks:
                        hook.pre_comm(step=S, level_number=0)
                    for hook in controller.hooks:
                        hook.post_comm(step=S, level_number=0, add_to_stats=True)
        print(f'{comm.rank} is at 3!', flush=True)
