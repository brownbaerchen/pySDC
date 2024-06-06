import numpy as np
import scipy

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class Heat1DChebychev(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, poly_coeffs=None, use_cheby_grid=True):
        self.poly_coeffs = poly_coeffs if poly_coeffs else [1, 2, 3, -4, -8, 19]

        assert use_cheby_grid

        # construct grids
        self.x = np.linspace(-1.3, 1.1, nvars)
        self.y = np.cos(np.pi / nvars * (np.arange(nvars) + 0.5))
        if use_cheby_grid:
            self.x = self.y
        self.inter_mat = self._get_interpolation_matrix()

        self.norm = np.ones(nvars) / nvars
        self.norm[0] /= 2

        # construct differentiation matrix
        self.L = np.array([np.polynomial.Chebyshev((0,) * (i) + (1,)).deriv(2)(self.x) for i in range(nvars)]).T
        self.Id = np.array([np.polynomial.Chebyshev((0,) * (i) + (1,)).deriv(0)(self.x) for i in range(nvars)]).T

        # register variables etc.
        self._makeAttributeAndRegister('nvars', 'use_cheby_grid', localVars=locals(), readOnly=True)
        super().__init__(init=(nvars, None, np.dtype('float64')))

    @property
    def _get_initial_polynomial(self):
        return np.polynomial.Polynomial(self.poly_coeffs)

    def _get_interpolation_matrix(self):
        """
        We use two grids: `x` for the grid we want to simulate and
        """
        from pySDC.core.Lagrange import LagrangeApproximation

        interp = LagrangeApproximation(points=self.x)
        return interp.getInterpolationMatrix(self.y)

    def _interpolate_to_chebychev_grid(self, u):
        return self.inter_mat @ u

    def eval_f(self, u, *args, **kwargs):
        from scipy import interpolate

        u_c = interpolate.interp1d(self.x, u)
        u_cheby = u_c(self.y)
        u_hat = scipy.fft.dct(u_cheby) * self.norm
        return self.L @ u_hat

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

        # construct system matrix with boundary conditions
        A = self.Id.copy()
        A[1:-1, :] -= factor * self.L[1:-1, :]

        # solve for the coefficients in the Chebychev basis
        res = np.linalg.solve(A, rhs)

        sol[:] = np.polynomial.Chebyshev(res)(self.x)
        return sol

    def u_exact(self, t=0):
        assert t == 0
        return self._get_initial_polynomial(self.x)
