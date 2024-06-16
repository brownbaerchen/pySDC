import numpy as np
import scipy
from scipy import sparse as sp

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh

from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper


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

        self.cheby = ChebychovHelper(N=nvars, S=self.S, mode=mode)
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
        zero = self.cheby.get_zero()
        Id = self.cheby.get_Id()
        D = self.cheby.get_differentiation_matrix()

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

        # prepare BCs
        if self.mode == 'D2U':
            bc_left = self.cheby.get_Dirichlet_BC_row_D(-1)
            bc_right = self.cheby.get_Dirichlet_BC_row_D(1)
        elif self.mode == 'T2U':
            bc_left = self.cheby.get_Dirichlet_BC_row_T(-1)
            bc_right = self.cheby.get_Dirichlet_BC_row_T(1)
        else:
            raise NotADirectoryError
        BC = (self.M * 0).tolil()
        BC[self.nvars - 1, : self.nvars] = bc_left
        BC[-1, : self.nvars] = bc_right

        self.BC_mask = BC != 0
        self.BCs = BC[self.BC_mask]
        BC[-1, self.nvars + 1 :] = 0  # not sure if we need this

        super().__init__(init=((self.S, nvars), None, np.dtype('float64')))

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        f[self.iu][:] = self.nu * self._compute_derivative(u[self.idu])
        return f

    def _compute_derivative(self, u):
        assert u.ndim == 1, u.shape
        u_hat = self.cheby.transform(u)
        return self.cheby.itransform(self.U2T @ self.D @ self.conv_inv @ u_hat)

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

        rhs_hat = self.cheby.transform(rhs)
        for i in range(self.S):
            rhs_hat[i] = self.conv_inv @ rhs_hat[i]

        _A = self.M + factor * self.L
        _rhs = self.M @ rhs_hat.flatten()

        # apply boundary conditions
        _A[self.BC_mask] = self.BCs
        # _A[-1, self.nvars + 1 :] = 0  # not sure if we need this
        _rhs[self.nvars - 1] = self.a
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
        sol[:] = self.cheby.itransform(sol_hat)
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

        me[self.idu] = self.cheby.itransform(self.U2T @ self.D @ coeffs)
        return me


class Heat2d(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nx=32,
        nz=32,
        nu=1.0,
        a=0.0,
        b=0.0,
    ):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        self.S = 2  # number of components in the solution
        self.iu = 0  # index for solution
        self.idu = 1  # index for first space derivative of solution

        super().__init__(init=((self.S, nx, nz), None, np.dtype('float64')))

        self.cheby = ChebychovHelper(N=nz, S=self.S, sparse_format='lil')
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
        self.Dz = self.cheby.get_T2U_differentiation_matrix()

        # setup Laplacian in x-direction
        k = np.fft.fftfreq(nx, 1.0 / nx)
        self.Dx = sp.diags(1j * k)

        # setup 2D operators
        self.Ix = sp.eye(nx, format='lil')
        self.Iz = self.T2U

        self.D = sp.kron(self.Ix, self.Dz) + sp.kron(self.Dx, self.Iz)
        # self.D = sp.kron(self.Ix, self.Dz)
        # self.D = sp.kron(self.Dx, self.Iz)
        zero = sp.kron(self.Ix, self.Iz) * 0
        Id = sp.kron(self.Ix, self.Iz) + sp.kron(self.Iz, self.Ix)

        # self.L = sp.bmat([[zero, -nu * self.D], [-self.D, Id]])
        self.M = sp.bmat([[Id, zero], [zero, zero]])

    def transform(self, u):
        u_fft = np.fft.fft(u, axis=0)
        u_hat = scipy.fft.dct(u_fft, axis=1) * self.norm
        return u_hat

    def itransform(self, u_hat):
        u_fft = np.fft.ifft(u_hat, axis=0)
        u = scipy.fft.idct(u_fft / self.norm, axis=1)
        return u.real

    def _compute_derivative(self, u):
        u_hat = self.transform(u)
        D_u_hat = self.D @ u_hat.flatten()
        return self.itransform(D_u_hat.reshape(u_hat.shape))

    def u_exact(self, *args, **kwargs):
        u = self.u_init
        # u[0][:] = np.sin(self.X) * np.sin(self.Z)
        u[0][:] = np.exp(-((self.X - np.pi) ** 2) - (np.pi * self.Z) ** 2)
        u[1][:] = self._compute_derivative(u[0])
        return u

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        f[self.iu][:] = self.nu * self._compute_derivative(u[self.idu])
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

        rhs_hat = self.transform(rhs)

        # apply boundary conditions
        bc_left = self.cheby.get_Dirichlet_BC_row_T(-1)
        bc_right = self.cheby.get_Dirichlet_BC_row_T(1)

        Az00 = self.Iz.copy()
        Az00[-1] = bc_left

        Az01 = -factor * self.nu * self.Dz

        Az10 = -factor * self.Dz
        Az10[-1] = bc_right

        Az11 = factor * self.Iz

        Az = sp.bmat([[Az00, Az10], [Az10, Az11]])

        Ax00 = self.Ix.copy()
        Ax01 = -factor * self.nu * self.Dx
        Ax10 = -factor * self.Dx
        Ax11 = factor * self.Ix

        Ax = sp.bmat([[Ax00, Ax10], [Ax10, Ax11]])

        # A = sp.kron(self.Ix, Az) + sp.kron(Ax, self.Iz)
        A = sp.kron(Ax, self.Iz)

        _rhs = self.M @ rhs_hat.flatten()
        # _rhs[self.nz-1:self.nx*self.nz:self.nz] = self.a
        # _rhs[(self.nx + 1)*self.nz-1::self.nz] = self.b

        # me = np.zeros_like(rhs_hat)
        # me[0][:,-1] = self.a
        # me[1][:,-1] = self.b
        # print(me)
        # print(me.flatten())
        # me2 = np.zeros_like(me).flatten()
        # me2[self.nz-1:self.nx*self.nz:self.nz] = self.a
        # me2[(self.nx + 1)*self.nz-1::self.nz] = self.b
        # print(me2)

        res = sp.linalg.spsolve(A, _rhs)

        sol_hat = res.reshape(sol.shape)
        sol[:] = self.itransform(sol_hat)
        return sol


class AdvectionDiffusion(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nx=32, nz=32, nu=1.0, a=0.0, b=0.0, c=1.0):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        self.S = 2  # number of components in the solution
        self.iu = 0  # index for solution
        self.idu = 1  # index for first space derivative of solution

        super().__init__(init=((self.S, nx, nz), None, np.dtype('float64')))

        # self.cheby = ChebychovHelper(N=nz, S=self.S, sparse_format='lil')
        # self.norm = self.cheby.get_norm()
        # self.T2U = self.cheby.get_conv('T2U')
        # self.T2D = self.cheby.get_conv('T2D')
        # self.D2T = self.cheby.get_conv('D2T')
        # self.U2T = self.cheby.get_conv('U2T')

        self.fft = FFTHelper(N=nx)
        self.cheby = ChebychovHelper(N=nz, mode='T2U')

        # generate grid
        self.x = self.fft.get_1dgrid()
        self.z = self.cheby.get_1dgrid()
        self.Z, self.X = np.meshgrid(self.z, self.x)

        # setup 1D operators
        self.Dz = self.cheby.get_differentiation_matrix()
        self.Dx = self.fft.get_differentiation_matrix()
        self.Ix = self.fft.get_Id()
        self.Iz = self.cheby.get_Id()

        self.D = sp.kron(self.Ix, self.Dz) + sp.kron(self.Dx, self.Iz)
        self.O = sp.kron(self.Ix, -self.nu * self.Dz) + sp.kron(-self.c * self.Ix, self.Iz)
        # self.O = sp.kron(self.Ix, self.Dz)
        # self.O = sp.kron(self.Dx, self.Iz)
        # self.O =  sp.kron(self.c * self.Ix, self.Iz)
        zero = sp.kron(self.Ix, self.Iz) * 0
        Id = sp.kron(self.Ix, self.Iz)
        self.U2T = sp.kron(self.Ix, self.cheby.get_conv('U2T'))

        print(self.D.toarray())

        self.L = sp.bmat([[zero, self.O], [-self.D, Id]])
        self.M = sp.bmat([[Id, zero], [zero, zero]])

        # apply boundary conditions
        bc_left = self.cheby.get_Dirichlet_BC_row_T(-1)
        bc_right = self.cheby.get_Dirichlet_BC_row_T(1)

        BCza = (self.Dz * 0).tolil()
        BCza[-1, :] = bc_left
        BCzb = (self.Dz * 0).tolil()
        BCzb[-1, :] = bc_right
        # BCz[self.nz - 1, : self.nz] = bc_left
        # BCz[-1, : self.nz] = bc_right
        _BCa = sp.kron(self.Ix, BCza, format='lil')
        _BCb = sp.kron(self.Ix, BCzb, format='lil')
        BC = sp.bmat([[_BCa, zero], [_BCb, zero]]).tolil()
        print(BC.toarray())

        self.bc_mask = BC != 0
        self.BCs = BC[self.bc_mask]

    def transform(self, u):
        _u_hat = self.fft.transform(u, axis=-2)
        u_hat = self.cheby.transform(_u_hat, axis=-1)
        return u_hat

    def itransform(self, u_hat):
        _u = self.fft.itransform(u_hat, axis=-2)
        u = self.cheby.itransform(_u, axis=-1)
        return u.real

    def _compute_derivative(self, u):
        assert u.ndim == 2, u.shape
        u_hat = self.transform(u)
        D_u_hat = self.U2T @ self.D @ u_hat.flatten()
        return self.itransform(D_u_hat.reshape(u_hat.shape))

    def u_exact(self, *args, **kwargs):
        u = self.u_init
        u[0][:] = np.sin(self.X) + np.sin(self.Z * np.pi) + (self.b - self.a) / 2 * self.Z + (self.b + self.a) / 2.0
        # u[0][:] = np.sin(self.Z * np.pi)
        # u[0][:] = self.Z**5 / 5
        # u[0][:] = np.exp(-(self.X-np.pi)**2 - (np.pi * self.Z)**2)
        u[1][:] = self._compute_derivative(u[0])
        return u

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        u_hat = self.transform(u[0])
        D_u_hat = self.O @ u_hat.flatten()
        f[:] = self.itransform(D_u_hat.reshape(u_hat.shape))
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

        rhs_hat = self.transform(rhs)

        A = self.M + factor * self.L
        _rhs = self.M @ rhs_hat.flatten()

        A[self.bc_mask] = self.BCs

        # _rhs[self.nvars - 1] = self.a
        _rhs[self.nz - 1 : self.nx * self.nz : self.nz] = self.a
        _rhs[(self.nx + 1) * self.nz - 1 :: self.nz] = self.b

        print(A.toarray())
        print(_rhs.real)

        res = sp.linalg.spsolve(A.tocsc(), _rhs)

        sol_hat = res.reshape(sol.shape)
        sol[:] = self.itransform(sol_hat)
        return sol
