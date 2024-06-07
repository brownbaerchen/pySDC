import numpy as np
import scipy

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class Heat1DChebychev(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, poly_coeffs=None):
        self._makeAttributeAndRegister('nvars', localVars=locals(), readOnly=True)
        self.poly_coeffs = poly_coeffs if poly_coeffs else [1, 2, 3, -4, -8, 19]

        # construct grids
        self.x = np.cos(np.pi / nvars * (np.arange(nvars) + 0.5))

        self.norm = np.ones(nvars) / nvars
        self.norm[0] /= 2

        self.D = self.get_differentiation_matrix(2)
        self.Id = np.diag(np.ones(nvars))

        super().__init__(init=(nvars, None, np.dtype('float64')))

    @property
    def _get_initial_polynomial(self):
        return np.polynomial.Polynomial(self.poly_coeffs)

    def get_differentiation_matrix(self, deriv):
        N = self.nvars
        D = np.zeros((N, N))
        for j in range(N):
            for k in range(j):
                D[k, j] = 2 * j * ((j - k) % 2)

        D[0, :] /= 2
        return np.linalg.matrix_power(D, deriv)

    def eval_f(self, u, *args, **kwargs):
        u_hat = scipy.fft.dct(u) * self.norm
        return scipy.fft.idct(self.D @ u_hat / self.norm)

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

        rhs_hat = scipy.fft.dct(rhs) * self.norm

        A = self.Id - factor * self.D
        sol_hat = np.linalg.solve(A, rhs_hat)
        sol[:] = scipy.fft.idct(sol_hat / self.norm)
        return sol

    def u_exact(self, t=0):
        assert t == 0
        return self._get_initial_polynomial(self.x)


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
