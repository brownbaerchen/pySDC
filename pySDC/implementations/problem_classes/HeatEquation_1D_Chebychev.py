import numpy as np
import scipy
from scipy import sparse as sp

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh

from pySDC.helpers.problem_helper import ChebychovHelper


class Heat1DChebychev(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, a=0, b=0, poly_coeffs=None):
        self._makeAttributeAndRegister('nvars', 'a', 'b', localVars=locals(), readOnly=True)
        self.poly_coeffs = poly_coeffs if poly_coeffs else [1, 2, 3, -4, -8, 19]

        cheby = ChebychovHelper(N=nvars)
        self.T2D = cheby.get_T2D()
        self.D2T = cheby.get_D2T()
        self.T2U = cheby.get_T2U()
        self.U2T = cheby.get_U2T()

        self.iu = 0  # index for solution
        self.idu = 1  # index for first space derivative of solution

        # construct grids
        self.x = cheby.get_1dgrid()
        self.norm = cheby.get_norm()

        # setup matrices for derivative / integration
        D = cheby.get_T2U_differentiation_matrix()
        zero = sp.eye(self.nvars) * 0.0

        self.L = sp.bmat([[zero, -D], [-D, self.T2U]])
        self.M = sp.bmat([[self.T2U, zero], [zero, zero]])
        self.D = D

        super().__init__(init=((2, nvars), None, np.dtype('float64')))

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init

        u_hat = scipy.fft.dct(u, axis=1) * self.norm

        # f[self.idu][:] = scipy.fft.idct(self.U2T @ self.D @ u_hat[self.iu] / self.norm)
        f[self.iu][:] = scipy.fft.idct(self.U2T @ self.D @ u_hat[self.idu] / self.norm)
        return f

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

        A = self.M + factor * self.L
        # print(self.M.toarray())
        # print(self.L.toarray())
        # print(A.toarray())
        # print(self.M@rhs_hat.flatten())

        sol_hat = sp.linalg.spsolve(A, self.M @ rhs_hat.flatten()).reshape(sol.shape)
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
        D_coeffs = self.T2D @ coeffs
        D_coeffs[0] = (self.b + self.a) / 2
        D_coeffs[1] = (self.b - self.a) / 2
        coeffs = self.D2T @ D_coeffs
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
