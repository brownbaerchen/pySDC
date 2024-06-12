import numpy as np
import scipy
from scipy import sparse as sp

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh

from pySDC.helpers.problem_helper import ChebychovHelper


class Heat1DChebychev(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, a=0, b=0, poly_coeffs=None, solver_type='direct', lintol=1e-9):
        self._makeAttributeAndRegister('nvars', 'a', 'b', 'solver_type', 'lintol', localVars=locals(), readOnly=True)
        self.poly_coeffs = poly_coeffs if poly_coeffs else [1, 2, 3, -4, -8, 19]

        cheby = ChebychovHelper(N=nvars)
        self.T2D = cheby.get_conv('T2D')
        self.D2T = cheby.get_conv('D2T')
        self.T2U = cheby.get_conv('T2U')
        self.U2T = cheby.get_conv('U2T')
        self.U2D = cheby.get_conv('U2D')
        self.D2U = cheby.get_conv('D2U')

        S = 2  # number of components in the solution
        self.iu = 0  # index for solution
        self.idu = 1  # index for first space derivative of solution

        # construct grid
        self.x = cheby.get_1dgrid()
        self.norm = cheby.get_norm()

        # setup operators between components going from T to U
        zero = sp.eye(self.nvars) * 0.0
        Id = sp.eye(self.nvars) @ self.D2U
        D = cheby.get_T2U_differentiation_matrix()

        # Adapt derivative matrix such that it does not change the boundary conditions
        D_D = (D @ self.U2D).tolil()
        D_D[0, :] = 0
        D_D[1, :] = 0
        D = D_D @ self.D2U
        self.D = D

        # setup preconditioner to generate a banded matrix
        n = np.arange(nvars)
        perm = np.stack((n, n + nvars), axis=1).flatten()
        Pl = sp.eye(S * nvars).toarray()
        self.Pl = sp.csc_matrix(Pl[perm])
        self.Pl = sp.eye(S * nvars)
        self.Pr = sp.linalg.inv(self.Pl)

        # setup system matrices connecting the components
        self.L = sp.bmat([[zero, -D], [-D, Id]])
        self.M = sp.bmat([[Id, zero], [zero, zero]])

        super().__init__(init=((S, nvars), None, np.dtype('float64')))

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        f[self.iu][:] = self._compute_derivative(u[self.idu])
        return f

    def _compute_derivative(self, u):
        u_hat = scipy.fft.dct(u) * self.norm
        return scipy.fft.idct(self.U2T @ self.D @ u_hat / self.norm)

    def _apply_BCs(self, u_hat):
        u_D = self.T2D @ u_hat
        u_D[0] = (self.b + self.a) / 2
        u_D[1] = (self.b - self.a) / 2
        return self.D2T @ u_D

    def solve_system(self, rhs, factor, *args, **kwargs):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.

        Returns
        -------
        sol : dtype_u
            The solution of the linear solver.
        """
        sol = self.u_init

        rhs_hat = scipy.fft.dct(rhs, axis=1) * self.norm
        rhs_hat[self.iu][:] = self._apply_BCs(rhs_hat[self.iu])

        A = self.Pl @ (self.M + factor * self.L) @ self.Pr
        _rhs = self.Pl @ self.M @ rhs_hat.flatten()

        print(A.toarray())

        # A_inv = sp.linalg.inv(A)
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3)
        # axs[0].imshow(np.log(abs(A.toarray())))
        # axs[1].imshow(np.log(abs(A_inv.toarray())))
        # axs[2].imshow(np.log(abs((A_inv@A).toarray())))
        # plt.show()

        if self.solver_type == 'direct':
            res = sp.linalg.spsolve(A, _rhs)
        elif self.solver_type == 'gmres':
            res, _ = sp.linalg.gmres(A, _rhs, tol=self.lintol)
        else:
            raise NotImplementedError(f'Solver {self.solver_type!r} is not implemented')

        sol_hat = (self.Pr @ res).reshape(sol.shape)
        sol[:] = scipy.fft.idct(sol_hat / self.norm, axis=1)
        return sol

    def u_exact(self, t=0):
        """
        This will use the polynomial coefficients supplied, but will change it to enforce the BC's.

        Returns:
            pySDC.dataype.mesh: Initial conditions
        """
        assert t == 0
        me = self.u_init

        coeffs = np.zeros(self.nvars)
        coeffs[: len(self.poly_coeffs)] = self.poly_coeffs
        coeffs = self._apply_BCs(coeffs)
        me[self.iu][:] = np.polynomial.Chebyshev(coeffs)(self.x)

        me[self.idu] = scipy.fft.idct(self.U2T @ self.D @ coeffs / self.norm)
        return me


class Heat2d(ptype):
    def __init__(self, nx=32, nz=32):
        super().__init__(init=((nx, nz), None, np.dtype('float64')))

        self._makeAttributeAndRegister('nx', 'nz', localVars=locals(), readOnly=True)

        # generate grid
        self.x = 2 * np.pi / (nx + 1) * np.arange(nx)
        self.z = np.cos(np.pi / nz * (np.arange(nz) + 0.5))
        self.X, self.Z = np.meshgrid(self.x, self.z)

        # setup Laplacian in x-direction
        k = np.fft.fftfreq(nx, 1.0 / nx)

    def u_exact(self, *args, **kwargs):
        return np.sin(self.X) * np.cos(self.Z)
