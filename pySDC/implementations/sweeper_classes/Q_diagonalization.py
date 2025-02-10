from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
import numpy as np
import scipy.sparse as sp


class QDiagonalization(generic_implicit):
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
        if 'update_f_evals' not in params.keys():
            params['update_f_evals'] = False
        if 'ignore_ic' not in params.keys():
            params['ignore_ic'] = False

        if not params.get('ignore_ic', False):
            params['initial_guess'] = 'zero'

        super().__init__(params)

        self.set_G_inv(self.params.G_inv)

    def set_G_inv(self, G_inv):
        self.params.G_inv = G_inv
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
        """
        Compute matrix-vector multiplication. Vector can be list.

        Args:
            mat: Matrix
            vec: Vector

        Returns:
            list: mat @ vec
        """
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

        if L.tau[0] is not None:
            raise NotImplementedError('This sweeper does not work with multi-level SDC')

        # perform local solves on the collocation nodes, can be parallelized!
        if self.params.ignore_ic:
            x1 = self.mat_vec(self.S_inv, [self.level.u[m + 1] for m in range(M)])
        else:
            x1 = self.mat_vec(self.S_inv, [self.level.u[0] for _ in range(M)])
        x2 = []
        for m in range(M):
            x2.append(
                P.solve_system(rhs=x1[m], factor=self.w[m] * L.dt, u0=x1[m], t=L.time + L.dt * self.coll.nodes[m])
            )
        z = self.mat_vec(self.S, x2)
        y = self.mat_vec(self.params.G_inv, z)

        # update solution and evaluate right hand side
        for m in range(M):
            L.u[m + 1] = y[m]
            if self.params.update_f_evals:
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        L.status.updated = True
        return None

    def eval_f_at_all_nodes(self):
        L = self.level
        P = self.level.prob
        for m in range(self.coll.num_nodes):
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])
