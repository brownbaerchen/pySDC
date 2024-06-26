import numpy as np

from pySDC.core.problem import Problem
from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class RayleighBenard(Problem):

    dtype_u = mesh
    dtype_f = imex_mesh
    xp = np

    def __init__(self, Pr=1, Ra=1, nx=8, nz=8, cheby_mode='T2T', BCs=None, **kwargs):
        BCs = {} if BCs is None else BCs
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        S = 8  # number of variables

        super().__init__(init=((S, nx, nz), None, np.dtype('complex128')))

        self.indices = {
            'u': 0,
            'v': 1,
            'ux': 2,
            'vz': 3,
            'T': 4,
            'Tx': 5,
            'Tz': 6,
            'p': 7,
        }
        self.index_to_name = {v: k for k, v, in self.indices.items()}

        # prepare indexes for the different components. Do this by hand so that the editor can autocomplete...
        self.iu = self.indices['u']  # velocity in x-direction
        self.iv = self.indices['v']  # velocity in z-direction
        self.iux = self.indices['ux']  # derivative of u wrt x
        self.ivz = self.indices['vz']  # derivative of v wrt z
        self.iT = self.indices['T']  # temperature
        self.iTx = self.indices['Tx']  # derivative of temperature wrt x
        self.iTz = self.indices['Tz']  # derivative of temperature wrt z
        self.ip = self.indices['p']  # pressure

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
        Sx1D = self.fft.get_integration_matrix()
        Sz1D = self.cheby.get_integration_matrix()

        # construct 2D matrices
        Dx = sp.kron(Dx1D, Iz1D)
        Dz = sp.kron(Ix1D, Dz1D)
        I = sp.kron(Ix1D, Iz1D)
        O = I * 0
        self.Dx = Dx
        self.Dz = Dz
        self.U2T = sp.kron(Ix1D, U2T1D)
        S2D = self.U2T @ sp.kron(Ix1D, Sz1D) @ sp.kron(Sx1D, sp.eye(nz))

        # construct operators
        L = self.cheby.get_empty_operator_matrix(S, O)
        M = self.cheby.get_empty_operator_matrix(S, O)

        # relations between quantities and derivatives
        L[self.iux][self.iu] = Dx.copy()
        L[self.iux][self.iux] = -I
        L[self.ivz][self.iv] = Dz.copy()
        L[self.ivz][self.ivz] = -I
        L[self.iTx][self.iT] = Dx.copy()
        L[self.iTx][self.iTx] = -I
        L[self.iTz][self.iT] = Dz.copy()
        L[self.iTz][self.iTz] = -I

        # divergence-free constraint
        L[self.ivz][self.ivz] = I.copy()
        L[self.ivz][self.iux] = -I.copy()

        # pressure gauge
        L[self.ip][self.ip] = S2D

        # differential terms
        L[self.iu][self.ip] = -Pr * Dx
        L[self.iu][self.iux] = Pr * Dx

        L[self.iv][self.ip] = -Pr * Dz
        L[self.iv][self.ivz] = Pr * Dz
        L[self.iv][self.iT] = Pr * Ra * I

        L[self.iT][self.iTx] = Dx
        L[self.iT][self.iTz] = Dz

        # mass matrix
        for i in [self.iu, self.iv, self.iT]:
            M[i][i] = I

        self.L = sp.bmat(L)
        self.M = sp.bmat(M)

        # prepare BCs
        BC_up = sp.eye(nz, format='lil') * 0
        BC_up[-1, :] = self.cheby.get_Dirichlet_BC_row_T(1)
        BC_up = sp.kron(Ix1D, BC_up, format='lil')

        BC_down = sp.eye(nz, format='lil') * 0
        BC_down[-1, :] = self.cheby.get_Dirichlet_BC_row_T(-1)
        BC_down = sp.kron(Ix1D, BC_down, format='lil')

        # TODO: distribute tau terms sensibly
        # self.iub = self.ip
        # self.iua = self.iu
        BC = self.cheby.get_empty_operator_matrix(S, O)
        # BC[self.iua][self.iu] = BC_up
        # BC[self.iub][self.iu] = BC_down
        BC[self.ip][self.ip] = BC_up
        BC[self.iv][self.iv] = BC_up
        BC[self.ivz][self.iv] = BC_down
        BC[self.iT][self.iT] = BC_up
        BC[self.iTz][self.iT] = BC_down

        BC = sp.bmat(BC, format='lil')
        self.BC_mask = BC != 0
        self.BC = BC[self.BC_mask]

        # prepare mask to zero rows that we put BCs into
        rhs_BC_hat = self.u_init
        # rhs_BC_hat[self.iua, :, -1] = 1
        # rhs_BC_hat[self.iub, :, -1] = 1
        rhs_BC_hat[self.iv, :, -1] = 1
        rhs_BC_hat[self.ip, :, -1] = 1
        rhs_BC_hat[self.ivz, :, -1] = 1
        rhs_BC_hat[self.iT, :, -1] = 1
        rhs_BC_hat[self.iTz, :, -1] = 1
        mask = rhs_BC_hat.flatten() == 1
        self.BC_zero_idx = self.xp.arange(self.nx * self.nz * S)[mask]

    def transform(self, u):
        assert u.ndim > 1, 'u must not be flattened here!'
        u_hat = u * 0
        _u_hat = self.fft.transform(u, axis=-2)
        u_hat[:] = self.cheby.transform(_u_hat, axis=-1)
        return u_hat

    def itransform(self, u_hat):
        assert u_hat.ndim > 1, 'u_hat must not be flattened here!'
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

        # linear temperature gradient
        Ttop = 1.0
        Tbottom = 0.0
        me[self.iT] = (Ttop - Tbottom) / 2 * self.Z + (Ttop + Tbottom) / 2.0

        # perturb slightly
        me[self.iT] += self.xp.random.rand(*me[self.iT].shape) * 1e-3

        # evaluate derivatives
        derivatives = self._compute_derivatives(me)
        for i in [self.iTx, self.iTz, self.iux, self.ivz]:
            me[i] = derivatives[i]

        return me

    def _put_BCs_in_rhs(self, rhs):
        assert rhs.ndim > 1, 'Right hand side must not be flattened here!'

        _rhs_hat = self.cheby.transform(rhs, axis=-1)

        # _rhs_hat[self.iua, :, -1] = self.BCs.get('u_top', 0)
        # _rhs_hat[self.iub, :, -1] = self.BCs.get('u_bottom', 0)
        _rhs_hat[self.iv, :, -1] = self.BCs.get('v_top', 0)
        _rhs_hat[self.ivz, :, -1] = self.BCs.get('v_bottom', 0)
        _rhs_hat[self.iT, :, -1] = self.BCs.get('T_top', 1)
        _rhs_hat[self.iTz, :, -1] = self.BCs.get('T_bottom', 0)
        _rhs_hat[self.ip, :, -1] = self.BCs.get('p_top', 0)

        return self.cheby.itransform(_rhs_hat, axis=-1)

    def _put_BCs_in_matrix(self, A):
        A = A.tolil()
        A[self.BC_zero_idx, :] = 0
        A[self.BC_mask] = self.BC
        return A.tocsc()

    def solve_system(self, rhs, factor, *args, **kwargs):
        sol = self.u_init

        _rhs = self._put_BCs_in_rhs(rhs)
        rhs_hat = self.transform(_rhs)

        A = self.M + factor * self.L
        A = self._put_BCs_in_matrix(A)

        sol_hat = self.cheby.sparse_lib.linalg.spsolve(A.tocsc(), rhs_hat.flatten()).reshape(sol.shape)

        sol[:] = self.itransform(sol_hat)
        return sol

    def compute_vorticiy(self, u):
        u_hat = self.transform(u)
        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx
        vorticity_hat = (Dx * u_hat[self.iv].flatten() + Dz @ u_hat[self.iu].flatten()).reshape(u[self.iu].shape)
        return self.itransform(vorticity_hat)

    def get_fig(self):  # pragma: no cover
        """
        Get a figure suitable to plot the solution of this problem

        Returns
        -------
        self.fig : matplotlib.pyplot.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plt.rcParams['figure.constrained_layout.use'] = True
        self.fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=((8, 3)))
        self.cax = []
        divider = make_axes_locatable(axs[0])
        self.cax += [divider.append_axes('right', size='3%', pad=0.03)]
        divider2 = make_axes_locatable(axs[1])
        self.cax += [divider2.append_axes('right', size='3%', pad=0.03)]
        return self.fig

    def plot(self, u, t=None, fig=None):  # pragma: no cover
        r"""
        Plot the solution. Please supply a figure with the same structure as returned by ``self.get_fig``.

        Parameters
        ----------
        u : dtype_u
            Solution to be plotted
        t : float
            Time to display at the top of the figure
        fig : matplotlib.pyplot.figure.Figure
            Figure with the correct structure

        Returns
        -------
        None
        """
        fig = self.get_fig() if fig is None else fig
        axs = fig.axes

        vmin = u.min()
        vmax = u.max()

        imT = axs[0].pcolormesh(self.X, self.Z, u[self.iT].real)
        imV = axs[1].pcolormesh(self.X, self.Z, self.compute_vorticiy(u).real)

        for i, label in zip([0, 1], [r'$T$', 'vorticity']):
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2e}')
        axs[1].set_xlabel(r'$x$')
        axs[1].set_ylabel(r'$z$')
        fig.colorbar(imT, self.cax[0])
        fig.colorbar(imV, self.cax[1])
