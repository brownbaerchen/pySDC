import numpy as np

from pySDC.core.Problem import ptype
from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class RayleighBenard(ptype):

    dtype_u = mesh
    dtype_f = imex_mesh
    xp = np

    def __init__(self, Pr=1, Ra=1, nx=8, nz=8, cheby_mode='T2T', BCs=None, **kwargs):
        BCs = {} if BCs is None else BCs
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        S = 8  # number of variables

        # prepare indexes for the different components
        self.iu = 0  # velocity in x-direction
        self.iv = 1  # velocity in z-direction
        self.iux = 2  # derivative of u wrt x
        self.ivz = 3  # derivative of v wrt z
        self.iT = 4  # temperature
        self.iTx = 5  # derivative of temperature wrt x
        self.iTz = 6  # derivative of temperature wrt z
        self.ip = 7  # pressure

        self.cheby = ChebychovHelper(N=nz, mode=cheby_mode)
        self.fft = FFTHelper(N=nx)
        sp = self.cheby.sparse_lib

        # construct grid
        x = self.fft.get_1dgrid()
        z = self.cheby.get_1dgrid()
        self.Z, self.X = np.meshgrid(z, x)

        # construct 1D matrices
        Dx1D = self.fft.get_differentiation_matrix()
        Ix1D = self.fft.get_Id()
        Dz1D = self.cheby.get_differentiation_matrix()
        Iz1D = self.cheby.get_Id()
        T2U1D = self.cheby.get_conv(cheby_mode)
        U2T1D = self.cheby.get_conv(cheby_mode[::-1])

        # construct 2D matrices
        Dx = sp.kron(Dx1D, Iz1D)
        Dz = sp.kron(Ix1D, Dz1D)
        I = sp.kron(Ix1D, Iz1D)
        O = I * 0
        self.Dx = Dx
        self.Dz = Dz
        self.U2T = sp.kron(Ix1D, U2T1D)

        # construct operators
        L = self.cheby.get_empty_operator_matrix(S, O)
        M = self.cheby.get_empty_operator_matrix(S, O)

        # relations between quantities and derivatives
        L[self.iux][self.iu] = Dx.copy()
        L[self.ivz][self.iv] = Dz.copy()
        L[self.iTx][self.iT] = Dx.copy()
        L[self.iTz][self.iT] = Dz.copy()

        # divergence-free constraint
        L[self.iux][self.ivz] = -I.copy()
        L[self.ivz][self.iux] = -I.copy()  # do we need/want this?

        # differential terms
        L[self.iu][self.ip] = Pr * Dx
        L[self.iu][self.iux] = -Pr * Dx

        L[self.iv][self.ip] = Pr * Dz
        L[self.iv][self.ivz] = -Pr * Dz
        L[self.iv][self.iT] = -Pr * Ra * I

        L[self.iT][self.iTx] = -Dx
        L[self.iT][self.iTz] = -Dz

        # mass matrix
        for i in [self.iu, self.iv, self.iT]:
            M[i][i] = I

        self.L = sp.bmat(L)
        self.M = sp.bmat(M)

        # prepare BCs
        BC_up = sp.eye(nz, format='lil') * 0
        BC_up[-1, :] = self.cheby.get_Dirichlet_BC_row_T(-1)
        BC_up = sp.kron(Ix1D, BC_up, format='lil')

        BC_down = sp.eye(nz, format='lil') * 0
        BC_down[-1, :] = self.cheby.get_Dirichlet_BC_row_T(1)
        BC_down = sp.kron(Ix1D, BC_down, format='lil')

        # TODO: distribute tau terms sensibly
        BC = self.cheby.get_empty_operator_matrix(S, O)
        BC[self.iTz][self.iu] = BC_up
        BC[self.iux][self.iu] = BC_down
        BC[self.iT][self.iv] = BC_up
        BC[self.ivz][self.iv] = BC_down
        BC[self.iTx][self.iT] = BC_up
        BC[self.ip][self.iT] = BC_down

        BC = sp.bmat(BC, format='lil')
        self.BC_mask = BC != 0
        self.BC = BC[self.BC_mask]

        super().__init__(init=((S, nx, nz), None, np.dtype('complex128')))

    def transform(self, u):
        u_hat = u * 0
        _u_hat = self.fft.transform(u, axis=-2)
        u_hat[:] = self.cheby.transform(_u_hat, axis=-1)
        return u_hat

    def itransform(self, u_hat):
        u = u_hat * 0
        _u = self.fft.itransform(u_hat, axis=-2)
        u[:] = self.cheby.itransform(_u, axis=-1)
        return u

    def _compute_derivatives(self, u, skip_transform=False):
        me_hat = self.u_init

        u_hat = u if skip_transform else self.transform(u)
        shape = u[self.iu].shape
        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx

        for i, iD, D in zip(
            [self.iu, self.iv, self.iT, self.iT], [self.iux, self.ivz, self.iTx, self.iTz], [Dx, Dz, Dx, Dz]
        ):
            me_hat[iD][:] = (D @ u_hat[i].flatten()).reshape(shape)

        return me_hat if skip_transform else self.itransform(me_hat)

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init

        u_hat = self.transform(u)
        f_hat = self.f_init

        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx

        # evaluate implicit terms
        shape = u[self.iu].shape
        f_hat.impl[self.iT] = (Dx @ u_hat[self.iTx].flatten() + Dz @ u_hat[self.iTz].flatten()).reshape(shape)
        f_hat.impl[self.iu] = (-Dx @ u_hat[self.ip].flatten() + Dx @ u_hat[self.iux].flatten()).reshape(shape)
        f_hat.impl[self.iv] = (-Dz @ u_hat[self.ip].flatten() + Dz @ u_hat[self.ivz].flatten()).reshape(
            shape
        ) + self.Ra * u_hat[self.iT]

        f[:] = self.itransform(f_hat)

        # treat convection explicitly
        f.expl[self.iu] = -u[self.iu] * (u[self.iux] + u[self.ivz])
        f.expl[self.iv] = -u[self.iv] * (u[self.iux] + u[self.ivz])
        f.expl[self.iT] = -u[self.iu] * u[self.iTx] - u[self.iv] * u[self.iTz]

        return f

    def u_exact(self, t=0):
        assert t == 0

        me = self.u_init

        # fill everything with random noise
        me[:] = self.xp.random.rand(*me.shape) * 1e-3

        # linear temperature gradient
        Ttop = 1.0
        Tbottom = 0.0
        me[self.iT] = (Ttop - Tbottom) / 2 * self.Z + (Ttop + Tbottom) / 2.0

        # evaluate derivatives
        derivatives = self._compute_derivatives(me)
        for i in [self.iTx, self.iTz, self.iux, self.ivz]:
            me[i] = derivatives[i]

        return me

    def _put_BCs_in_rhs(self, rhs):
        assert rhs.ndim > 1, 'Right hand side must not be flattened here!'

        # _rhs_hat = (self.T2U @ self.cheby.transform(rhs, axis=-1).flatten()).reshape(rhs.shape)
        _rhs_hat = self.cheby.transform(rhs, axis=-1)

        _rhs_hat[self.iu, :, -1] = self.BCs.get('u_top', 0)
        _rhs_hat[self.iux, :, -1] = self.BCs.get('u_bottom', 0)
        _rhs_hat[self.iv, :, -1] = self.BCs.get('v_top', 0)
        _rhs_hat[self.ivz, :, -1] = self.BCs.get('v_bottom', 0)
        _rhs_hat[self.iTx, :, -1] = self.BCs.get('T_top', 0)
        _rhs_hat[self.ip, :, -1] = self.BCs.get('T_bottom', 0)

        return self.cheby.itransform(_rhs_hat, axis=-1)

    def _put_BCs_in_matrix(self, A):
        A = A.tolil()
        A[self.BC_mask] = self.BC
        return A.tocsc()

    def solve_system(self, rhs, factor, *args, **kwargs):
        sol = self.u_init

        # _rhs = (self.T2U @ self.cheby.transform(rhs, axis=-1).flatten()).reshape(rhs.shape)
        # _rhs[0, :, -1] = self.bc_lower
        # _rhs[1, :, -1] = self.bc_upper
        # rhs_hat = self.fft.transform(_rhs, axis=-2)

        # A = (self.M + factor * self.L).tolil()
        # A[self.BC != 0] = self.BC[self.BC != 0]

        # res = sp.linalg.spsolve(A.tocsc(), rhs_hat.flatten())

        # sol_hat = res.reshape(sol.shape)
        # sol[:] = self.itransform(sol_hat)
        return sol
