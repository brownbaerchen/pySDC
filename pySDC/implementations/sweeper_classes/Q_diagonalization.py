from pySDC.core.sweeper import Sweeper
import numpy as np


class QDiagonalization(Sweeper):
    """
    Sweeper solving the collocation problem directly via diagonalization of Q
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """
        if 'G_inv' not in params.keys():
            params['G_inv'] = np.eye(params['num_nodes'])

        super().__init__(params)

        self.w, self.S, self.S_inv = self.computeDiagonalization(A=self.coll.Qmat[1:, 1:] @ self.params.G_inv)

    @staticmethod
    def computeDiagonalization(A):
        """
        Compute diagonalization of dense matrix A = S diag(w) S^-1

        Args:
            A (numpy.ndarray): dense matrix to diagonalize

        Returns:
            numpy.array: Diagonal entries of the diagonalized matrix w
            numpy.ndarray: Matrix of eigenvectors S
            numpy.ndarray: Inverse of S
        """
        w, S = np.linalg.eig(A)
        S_inv = np.linalg.inv(S)
        assert np.allclose(S @ np.diag(w) @ S_inv, A)
        return w, S, S_inv

    def mat_vec(self, mat, vec):
        assert mat.shape[1] == len(vec)
        result = []
        for m in range(mat.shape[0]):
            result.append(self.level.prob.u_init)
            for j in range(mat.shape[1]):
                result[-1] += mat[m, j] * vec[j]
        return result

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        # only if the level has been touched before
        assert L.status.unlocked
        if L.tau[0] is not None:
            raise NotImplementedError('This sweeper does not work with multi-level SDC yet')

        # perform local solves on the collocation nodes in parallel
        x1 = self.mat_vec(self.S_inv, self.level.u[1:])
        x2 = []
        for m in range(M):
            x2.append(P.solve_system(rhs=x1[m], factor=self.w[m] * L.dt, u0=x1[m], t=L.time))
        z = self.mat_vec(self.S, x2)
        y = self.mat_vec(self.params.G_inv, z)

        for m in range(M):
            L.u[m + 1] = y[m]

        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * L.f[m + 1]
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
