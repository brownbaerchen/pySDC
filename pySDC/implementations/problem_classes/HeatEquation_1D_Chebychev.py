import numpy as np
import scipy
from scipy import sparse as sp

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh

from pySDC.helpers.problem_helper import ChebychovHelper


class Heat1DChebychev(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=128,
        a=0,
        b=0,
        poly_coeffs=None,
        solver_type='direct',
        lintol=1e-9,
        mode='D2U',
        nu=1.0,
        preconditioning=False,
    ):
        poly_coeffs = poly_coeffs if poly_coeffs else [1, 2, 3, -4, -8, 19]
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        self.S = 2  # number of components in the solution
        self.iu = 0  # index for solution
        self.idu = 1  # index for first space derivative of solution

        self.cheby = ChebychovHelper(N=nvars, S=self.S)
        self.T2U = self.cheby.get_conv('T2U')
        self.T2D = self.cheby.get_conv('T2D')
        self.D2T = self.cheby.get_conv('D2T')
        self.U2T = self.cheby.get_conv('U2T')

        if mode == 'T2U':
            self.conv = self.cheby.get_conv('T2T')
            self.conv_inv = self.cheby.get_conv('T2T')
        elif mode == 'D2U':
            self.conv = self.cheby.get_conv('D2T')
            self.conv_inv = self.cheby.get_conv('T2D')
        else:
            raise NotImplementedError

        # construct grid
        self.x = self.cheby.get_1dgrid()

        # setup operators between components going from T to U
        zero = sp.eye(self.nvars) * 0.0
        Id = sp.eye(self.nvars) @ self.T2U @ self.conv
        D = self.cheby.get_T2U_differentiation_matrix() @ self.conv

        self.D = D
        self.zero = zero
        self.Id = Id

        # setup preconditioner to generate a banded matrix
        if preconditioning:
            Pl = np.diag(np.ones(self.S * nvars))
            n = np.arange(nvars)
            perm = np.stack((n, n + nvars), axis=1).flatten()
            self.Pl = sp.csc_matrix(Pl[perm])
        else:
            self.Pl = sp.eye(self.S * nvars)
        self.Pr = sp.linalg.inv(self.Pl)

        # setup system matrices connecting the components
        self.L = sp.bmat([[zero, -nu * D], [-D, Id]])
        self.M = sp.bmat([[Id, zero], [zero, zero]])

        super().__init__(init=((self.S, nvars), None, np.dtype('float64')))

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        f[self.iu][:] = self.nu * self._compute_derivative(u[self.idu])
        return f

    def _compute_derivative(self, u):
        u_hat = self.cheby.dct(u, S=1)
        return self.cheby.idct(self.U2T @ self.D @ self.conv_inv @ u_hat, S=1)

    def _apply_BCs(self, u_hat):
        u_D = self.T2D @ u_hat
        u_D[0] = (self.b + self.a) / 2
        u_D[1] = (self.b - self.a) / 2
        return self.D2T @ u_D

    def solve_system(self, rhs, factor, u0, *args, **kwargs):
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

        rhs_hat = self.cheby.dct(rhs)
        for i in range(self.S):
            rhs_hat[i] = self.conv_inv @ rhs_hat[i]

        _A = self.M + factor * self.L
        _rhs = self.M @ rhs_hat.flatten()

        # apply boundary conditions
        if self.mode == 'D2U':
            bc_left = self.cheby.get_Dirichlet_BC_row_D(-1)
            bc_right = self.cheby.get_Dirichlet_BC_row_D(1)
        elif self.mode == 'T2U':
            bc_left = self.cheby.get_Dirichlet_BC_row_T(-1)
            bc_right = self.cheby.get_Dirichlet_BC_row_T(1)
        else:
            raise NotADirectoryError
        _A[self.nvars - 1, : self.nvars] = bc_left
        _A[-1, : self.nvars] = bc_right

        _rhs[self.nvars - 1] = self.a

        _A[-1, self.nvars + 1 :] = 0
        _rhs[-1] = self.b

        A = self.Pl @ (_A) @ self.Pr
        _rhs = self.Pl @ _rhs

        # import matplotlib.pyplot as plt
        # A_inv = sp.linalg.inv(A)
        # fig, axs = plt.subplots(1, 3)
        # axs[0].imshow(np.log(abs(A.toarray())))
        # axs[1].imshow(np.log(abs(A_inv.toarray())))
        # axs[2].imshow(np.log(abs((A_inv @ A).toarray())))
        # plt.show()

        if self.solver_type == 'direct':
            res = sp.linalg.spsolve(A, _rhs)
        elif self.solver_type == 'gmres':
            res, _ = sp.linalg.gmres(A, _rhs, tol=self.lintol, u0=u0)
        else:
            raise NotImplementedError(f'Solver {self.solver_type!r} is not implemented')

        sol_hat = (self.Pr @ res).reshape(sol.shape)
        for i in range(self.S):
            sol_hat[i] = self.conv @ sol_hat[i]
        sol[:] = self.cheby.idct(sol_hat)
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

        me[self.idu] = self.cheby.idct(self.U2T @ self.D @ coeffs, S=1)
        return me


class Heat2d(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nx=32,
        nz=32,
    ):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        self.S = 2  # number of components in the solution
        self.iu = 0  # index for solution
        self.idu = 1  # index for first space derivative of solution

        super().__init__(init=((self.S, nx, nz), None, np.dtype('float64')))

        self.cheby = ChebychovHelper(N=nz, S=self.S)
        self.norm = self.cheby.get_norm()
        self.T2U = self.cheby.get_conv('T2U')
        self.T2D = self.cheby.get_conv('T2D')
        self.D2T = self.cheby.get_conv('D2T')
        self.U2T = self.cheby.get_conv('U2T')

        # generate grid
        self.x = 2 * np.pi / (nx + 1) * np.arange(nx)
        self.z = self.cheby.get_1dgrid()
        self.Z, self.X = np.meshgrid(self.z, self.x)

        # setup 1D operators between components going from T to U
        zero_z = sp.eye(nz) * 0.0
        Id_z = sp.eye(nz) @ self.T2U
        D_z = self.cheby.get_T2U_differentiation_matrix()

        # setup Laplacian in x-direction
        k = np.fft.fftfreq(nx, 1.0 / nx)
        D_x = sp.diags(-1j * k)

        # setup 2D operators
        self.D = sp.kron(sp.eye(nx), D_z) + sp.kron(D_x, sp.eye(nz))

    def transform(self, u):
        u_fft = np.fft.fft(u, axis=0)
        u_hat = scipy.fft.dct(u_fft, axis=1) * self.norm
        return u_hat

    def itransform(self, u_hat):
        u_fft = np.fft.ifft(u_hat, axis=0)
        u = scipy.fft.idct(u_fft / self.norm, axis=1)
        return u

    def u_exact(self, *args, **kwargs):
        u = self.u_init
        u[0][:] = np.sin(self.X) * np.cos(4 * self.Z)

        u_hat = self.transform(u[0])
        D_u_hat = self.D @ u_hat.flatten()
        u[1][:] = self.itransform(D_u_hat.reshape(u_hat.shape))
        return u
