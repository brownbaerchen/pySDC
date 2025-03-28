import numpy as np

from qmat.lagrange import LagrangeApproximation
from pySDC.core.convergence_controller import ConvergenceController


class EstimatePolynomialError(ConvergenceController):
    """
    Estimate the local error by using all but one collocation node in a polynomial interpolation to that node.
    While the converged collocation problem with M nodes gives a order M approximation to this point, the interpolation
    gives only an order M-1 approximation. Hence, we have two solutions with different order, and we know their order.
    That is to say this gives an error estimate that is order M. Keep in mind that the collocation problem should be
    converged for this and has order up to 2M. Still, the lower order method can be used for time step selection, for
    instance.
    If the last node is not the end point, we can interpolate to that node, which is an order M approximation and compare
    to the order 2M approximation we get from the extrapolation step.
    By default, we interpolate to the second to last node.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Args:
            controller (pySDC.Controller.controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate
        from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence

        sweeper_params = description['sweeper_params']
        num_nodes = sweeper_params['num_nodes']
        quad_type = sweeper_params['quad_type']

        defaults = {
            'control_order': -75,
            'estimate_on_node': num_nodes + 1 if quad_type == 'GAUSS' else num_nodes - 1,
            'rel_error': False,
            **super().setup(controller, params, description, **kwargs),
        }
        self.comm = description['sweeper_params'].get('comm', None)

        if self.comm:
            from mpi4py import MPI

            self.prepare_MPI_datatypes()
            self.MPI_SUM = MPI.SUM

        controller.add_hook(LogEmbeddedErrorEstimate)
        self.check_convergence = CheckConvergence.check_convergence

        if quad_type != 'GAUSS' and defaults['estimate_on_node'] > num_nodes:
            from pySDC.core.errors import ParameterError

            raise ParameterError(
                'You cannot interpolate with lower accuracy to the end point if the end point is a node!'
            )

        self.interpolation_matrix = None

        return defaults

    def reset_status_variables(self, *args, **kwargs):
        """
        Add variable for embedded error

        Returns:
            None
        """
        self.add_status_variable_to_level('error_embedded_estimate')
        self.add_status_variable_to_level('order_embedded_estimate')

    def matmul(self, A, b, xp=np):
        """
        Matrix vector multiplication, possibly MPI parallel.
        The parallel implementation performs a reduce operation in every row of the matrix. While communicating the
        entire vector once could reduce the number of communications, this way we never need to store the entire vector
        on any specific rank.

        Args:
            A (2d np.ndarray): Matrix
            b (list): Vector
            xp: Either numpy or cupy

        Returns:
            List: Axb
        """

        if self.comm:
            res = [A[i, 0] * b[0] if b[i] is not None else None for i in range(A.shape[0])]
            buf = b[0] * 0.0
            for i in range(0, A.shape[0]):
                index = self.comm.rank + (1 if self.comm.rank < self.params.estimate_on_node - 1 else 0)
                send_buf = (
                    (A[i, index] * b[index])
                    if self.comm.rank != self.params.estimate_on_node - 1
                    else xp.zeros_like(res[0])
                )
                self.comm.Allreduce(send_buf, buf, op=self.MPI_SUM)
                res[i] += buf
            return res
        else:
            return A @ xp.asarray(b)

    def get_interpolated_solution(self, L, xp):
        """
        Get the interpolated solution for numpy or cupy data types

        Args:
            u_vec (array): Vector of solutions
            prob (pySDC.problem): Problem
        """
        coll = L.sweep.coll

        u = [
            L.u[i].flatten() if L.u[i] is not None else L.u[i]
            for i in range(coll.num_nodes + 1)
            if i != self.params.estimate_on_node
        ]
        return self.matmul(self.interpolation_matrix, u, xp=xp)[0].reshape(L.prob.init[0])

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Estimate the error

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.Step.step): The current step

        Returns:
            None
        """

        if self.check_convergence(S):
            L = S.levels[0]
            coll = L.sweep.coll
            nodes = np.append(np.append(0, coll.nodes), 1.0)
            estimate_on_node = self.params.estimate_on_node

            if hasattr(L.u[0], 'xp'):
                xp = L.u[0].xp
            else:
                xp = np

            if self.interpolation_matrix is None:
                interpolator = LagrangeApproximation(
                    points=[nodes[i] for i in range(coll.num_nodes + 1) if i != estimate_on_node]
                )
                self.interpolation_matrix = xp.array(interpolator.getInterpolationMatrix([nodes[estimate_on_node]]))

            u_inter = self.get_interpolated_solution(L, xp)

            # compute end point if needed
            if estimate_on_node == len(nodes) - 1:
                if L.uend is None:
                    L.sweep.compute_end_point()
                high_order_sol = L.uend
                rank = 0
                L.status.order_embedded_estimate = coll.num_nodes + 1
            else:
                high_order_sol = L.u[estimate_on_node]
                rank = estimate_on_node - 1
                L.status.order_embedded_estimate = coll.num_nodes * 1

            rescale = float(abs(u_inter)) if self.params.rel_error else 1

            if self.comm:
                buf = np.array(abs(u_inter - high_order_sol) / rescale if self.comm.rank == rank else 0.0)
                self.comm.Bcast(buf, root=rank)
                L.status.error_embedded_estimate = float(buf)
            else:
                L.status.error_embedded_estimate = abs(u_inter - high_order_sol) / rescale

            self.debug(
                f'Obtained error estimate: {L.status.error_embedded_estimate:.2e} of order {L.status.order_embedded_estimate}',
                S,
            )

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check if we allow the scheme to solve the collocation problems to convergence.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if description['sweeper_params'].get('num_nodes', 0) < 2:
            return False, 'Need at least two collocation nodes to interpolate to one!'

        return True, ""


class EstimatePolynomialErrorFiredrake(EstimatePolynomialError):
    def matmul(self, A, b):
        """
        Matrix vector multiplication, possibly MPI parallel.
        The parallel implementation performs a reduce operation in every row of the matrix. While communicating the
        entire vector once could reduce the number of communications, this way we never need to store the entire vector
        on any specific rank.

        Args:
            A (2d np.ndarray): Matrix
            b (list): Vector

        Returns:
            List: Axb
        """

        if self.comm:
            res = [A[i, 0] * b[0] if b[i] is not None else None for i in range(A.shape[0])]
            buf = 0 * b[0]
            for i in range(0, A.shape[0]):
                index = self.comm.rank + (1 if self.comm.rank < self.params.estimate_on_node - 1 else 0)
                send_buf = (
                    (A[i, index] * b[index]) if self.comm.rank != self.params.estimate_on_node - 1 else 0 * res[0]
                )
                self.comm.Allreduce(send_buf, buf, op=self.MPI_SUM)
                res[i] += buf
            return res
        else:
            res = []
            for i in range(A.shape[0]):
                res.append(A[i, 0] * b[0])
                for j in range(1, A.shape[1]):
                    res[-1] += A[i, j] * b[j]

            return res

    def get_interpolated_solution(self, L):
        """
        Get the interpolated solution for Firedrake data types
        We are not 100% sure that you don't need to invert the mass matrix here, but should be fine.

        Args:
            u_vec (array): Vector of solutions
            prob (pySDC.problem): Problem
        """
        coll = L.sweep.coll

        u = [
            L.u[i] if L.u[i] is not None else L.u[i]
            for i in range(coll.num_nodes + 1)
            if i != self.params.estimate_on_node
        ]
        return L.prob.dtype_u(self.matmul(self.interpolation_matrix, u)[0])
        # return L.prob.invert_mass_matrix(self.matmul(self.interpolation_matrix, u)[0])
