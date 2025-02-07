import itertools
import copy as cp
import numpy as np
import dill

from pySDC.core.controller import ParaDiagController
from pySDC.core import step as stepclass
from pySDC.core.errors import ControllerError, CommunicationError
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
from pySDC.helpers.ParaDiagHelper import get_G_inv_matrix


class controller_ParaDiag_nonMPI(ParaDiagController):
    """

    ParaDiag controller, running serialized version

    """

    def __init__(self, num_procs, controller_params, description, alpha, linear=True):
        """
        Initialization routine for ParaDiag controller

        Args:
           num_procs: number of parallel time steps (still serial, though), can be 1
           controller_params: parameter set for the controller and the steps
           description: all the parameters to set up the rest (levels, problems, transfer, ...)
           alpha (float): alpha parameter for ParaDiag
           linear (bool): Whether the implicit part of the problem is linear or not
        """
        super().__init__(controller_params, description, alpha=alpha, linear=linear, useMPI=False, n_steps=num_procs)

        self.MS = []

        for l in range(num_procs):
            G_inv = get_G_inv_matrix(l, num_procs, alpha, description['sweeper_params'])
            description['sweeper_params']['G_inv'] = G_inv

            self.MS.append(stepclass.Step(description))

        self.base_convergence_controllers += [BasicRestarting.get_implementation(useMPI=False)]
        for convergence_controller in self.base_convergence_controllers:
            self.add_convergence_controller(convergence_controller, description)

        if self.params.dump_setup:
            self.dump_setup(step=self.MS[0], controller_params=controller_params, description=description)

        if len(self.MS[0].levels) > 1:
            raise NotImplementedError('This controller does not support multiple levels')

        for C in [self.convergence_controllers[i] for i in self.convergence_controller_order]:
            C.reset_buffers_nonMPI(self)
            C.setup_status_variables(self, MS=self.MS)

    def apply_matrix(self, mat):
        """
        Apply a matrix on the step level. Needs to be square. Puts the result back into the controller.

        Args:
            mat: square LxL matrix with L number of steps
        """
        L = len(self.MS)
        assert np.allclose(mat.shape, L)
        assert len(mat.shape) == 2

        level = self.MS[0].levels[0]
        M = level.sweep.params.num_nodes
        prob = level.prob

        # buffer for storing the result
        res = [
            None,
        ] * L

        # compute matrix-vector product
        for i in range(mat.shape[0]):
            res[i] = [prob.u_init for _ in range(M + 1)]
            for j in range(mat.shape[1]):
                for m in range(M + 1):
                    res[i][m] += mat[i, j] * self.MS[j].levels[0].u[m]

        # put the result in the "output"
        for i in range(mat.shape[0]):
            for m in range(M + 1):
                self.MS[i].levels[0].u[m] = res[i][m]

    def ParaDiag_block_residual(self):
        """
        Compute the residual of the composite collocation problem in ParaDiag

        Returns:
            float: Residual
        """
        prob = self.MS[0].levels[0].prob
        L = len(self.MS)

        # store initial conditions on the steps because we need to put them back in the end
        _u0 = [me.levels[0].u[0] for me in self.MS]

        # communicate initial conditions for computing the residual with p2p
        self.MS[0].levels[0].u[0] = prob.dtype_u(self.ParaDiag_block_u0)
        for l in range(L):
            self.MS[l].levels[0].sweep.compute_end_point()
            if l > 0:
                self.MS[l].levels[0].u[0] = prob.dtype_u(self.MS[l - 1].levels[0].uend)

            # reevaluate f after FFT
            self.MS[l].levels[0].sweep.eval_f_at_all_nodes()

        # compute residuals of local collocation problems (can do in parallel)
        residuals = []
        for l in range(L):
            level = self.MS[l].levels[0]
            level.sweep.compute_residual()
            residuals.append(level.status.residual)

        # put back initial conditions to continue with ParaDiag
        for l in range(L):
            self.MS[l].levels[0].u[0] = _u0[l]

        # compute global residual via "Reduce"
        return max(residuals)

    def ParaDiag_communication(self):
        """
        Communicate the solution to the last step back to the first step as required in ParaDiag
        """
        # TODO: add hooks like in other communication functions in the controller

        L = len(self.MS)
        # compute solution at the end of the interval (can do in parallel)
        for l in range(L):
            self.MS[l].levels[0].sweep.compute_end_point()

        # communicate initial conditions for next iteration (MPI ptp communication)
        if self.ParaDiag_linear:
            # for linear problems, we only need to communicate the contribution due to the alpha perturbation
            self.MS[0].levels[0].u[0] = self.ParaDiag_block_u0 - self.ParaDiag_alpha * self.MS[-1].levels[0].uend
        else:
            raise NotImplementedError('Communication for nonlinear ParaDiag is not yet implemented')

    def it_ParaDiag(self):
        # TODO: add hooks
        self.ParaDiag_communication()

        # weighted FFT in time (implement with MPI Reduce, not-parallel)
        self.FFT_in_time()

        # perform local solves of "collocation problems" on the steps (do in parallel)
        for S in self.MS:
            assert len(S.levels) == 1, 'Multi-level SDC not implemented in ParaDiag'
            S.levels[0].sweep.update_nodes()

        # inverse FFT in time (implement with MPI Reduce, not-parallel)
        self.iFFT_in_time()
