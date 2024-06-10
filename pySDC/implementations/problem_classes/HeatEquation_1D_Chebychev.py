import numpy as np
import scipy
from scipy import sparse as sp

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh

from pySDC.helpers.problem_helper import ChebychovHelper


class Heat1DChebychev(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, poly_coeffs=None):
        self._makeAttributeAndRegister('nvars', localVars=locals(), readOnly=True)
        self.poly_coeffs = poly_coeffs if poly_coeffs else [1, 2, 3, -4, -8, 19]

        cheby = ChebychovHelper(N=nvars)

        # construct grids
        self.x = cheby.get_1dgrid()
        self.norm = cheby.get_norm()

        # setup matrices for derivative / integration
        self.D = cheby.get_T2Tdifferentiation_matrix(2)
        self.Id = sp.eye(self.nvars, format='csc')

        # get conversion matrix between Chebychev and Dirichlet polynomials
        self.CtoD = sp.eye(self.nvars, format='csc') - sp.diags(np.ones(self.nvars - 2), offsets=-2, format='csc')
        self.DtoC = sp.linalg.inv(self.CtoD)

        super().__init__(init=(nvars, None, np.dtype('float64')))

    @property
    def _get_initial_polynomial(self):
        return np.polynomial.Polynomial(self.poly_coeffs)

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init

        u_hat = scipy.fft.dct(u) * self.norm
        f[:] = scipy.fft.idct(self.D @ u_hat / self.norm)
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

        rhs_hat = scipy.fft.dct(rhs) * self.norm

        A = self.Id - factor * self.D

        # A = (self.Id - factor * self.D) @ self.DtoC
        # sol_d = np.linalg.solve(A, rhs_hat)
        # sol_hat = self.DtoC @ sol_d

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
