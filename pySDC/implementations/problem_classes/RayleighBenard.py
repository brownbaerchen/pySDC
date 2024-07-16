import numpy as np

from pySDC.core.problem import Problem
from pySDC.helpers.spectral_helper import SpectralHelper, ChebychovHelper, FFTHelper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class RayleighBenard(Problem):

    dtype_u = mesh
    dtype_f = imex_mesh
    xp = np

    def __init__(self, Pr=1, Ra=1, nx=8, nz=8, cheby_mode='T2T', BCs=None):
        BCs = {} if BCs is None else BCs
        BCs = {
            'T_top': 1,
            'T_bottom': 0,
            'v_top': 0,
            'v_bottom': 0,
            'p_top': 0,
            **BCs,
        }
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        S = 8  # number of variables

        self.helper = SpectralHelper()
        self.helper.add_axis(base='fft', N=nx)
        self.helper.add_axis(base='cheby', N=nz, mode=cheby_mode)
        self.helper.add_component(['u', 'v', 'ux', 'vz', 'T', 'Tx', 'Tz', 'p'])
        self.helper.setup_fft()

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

        self.Z, self.X = self.helper.get_grid()

        # construct 2D matrices
        Dx = self.helper.get_differentiation_matrix(axes=(0,))
        Dz = self.helper.get_differentiation_matrix(axes=(1,))
        I = self.helper.get_Id()
        self.Dx = Dx
        self.Dz = Dz
        self.U2T = self.helper.get_basis_change_matrix()
        S2D = self.helper.get_integration_matrix(axes=(0, 1))

        # construct operators
        L = self.helper.get_empty_operator_matrix()
        M = self.helper.get_empty_operator_matrix()

        # relations between quantities and derivatives
        # L[self.iux][self.iu] = Dx.copy()
        # L[self.iux][self.iux] = -I
        # L[self.ivz][self.iv] = Dz.copy()
        # L[self.ivz][self.ivz] = -I
        L[self.ivz][self.iux] = -I  # TODO: where does this go?
        L[self.ivz][self.ivz] = I.copy()  # TODO: where does this go?
        # L[self.iTx][self.iT] = Dx.copy()
        # L[self.iTx][self.iTx] = -I
        # L[self.iTz][self.iT] = Dz.copy()
        # L[self.iTz][self.iTz] = -I
        self.helper.add_equation(L, 'ux', {'u': Dx, 'ux': -I})
        self.helper.add_equation(L, 'vz', {'v': Dz, 'vz': -I})
        self.helper.add_equation(L, 'Tx', {'T': Dx, 'Tx': -I})
        self.helper.add_equation(L, 'Tz', {'T': Dz, 'Tz': -I})

        # divergence-free constraint
        # L[self.ip][self.ivz] = I.copy()
        # L[self.ip][self.iux] = -I.copy()

        # pressure gauge
        # L[self.ip][self.ip] = S2D

        # differential terms
        # L[self.iu][self.ip] = -Pr * Dx
        # L[self.iu][self.iux] = Pr * Dx

        # L[self.iv][self.ip] = -Pr * Dz
        # L[self.iv][self.ivz] = Pr * Dz
        # L[self.iv][self.iT] = Pr * Ra * I

        # L[self.iT][self.iTx] = Dx
        # L[self.iT][self.iTz] = Dz

        self.helper.add_equation(L, 'u', {'p': -Pr * Dx, 'ux': Pr * Dx})
        self.helper.add_equation(L, 'v', {'p': -Pr * Dz, 'vz': Pr * Dz, 'T': Pr * Ra * I})
        self.helper.add_equation(L, 'T', {'Tx': Dx, 'Tz': Dz})
        self.helper.add_equation(L, 'p', {'p': S2D})

        # mass matrix
        for i in [self.iu, self.iv, self.iT]:
            M[i][i] = I

        self.L = self.helper.convert_operator_matrix_to_operator(L)
        self.M = self.helper.convert_operator_matrix_to_operator(M)

        # prepare BCs
        BC_up = self.helper.get_BC(1, 1)
        BC_down = self.helper.get_BC(1, -1)

        # TODO: distribute tau terms sensibly
        self.helper.add_BC(component='p', equation='p', axis=1, x=1, v=self.BCs['p_top'])
        self.helper.add_BC(component='v', equation='v', axis=1, x=1, v=self.BCs['v_top'])
        self.helper.add_BC(component='v', equation='vz', axis=1, x=-1, v=self.BCs['v_bottom'])
        self.helper.add_BC(component='T', equation='T', axis=1, x=1, v=self.BCs['T_top'])
        self.helper.add_BC(component='T', equation='Tz', axis=1, x=-1, v=self.BCs['T_bottom'])
        self.helper.setup_BCs()

        self.BC = self.helper._BCs
        self.BC_zero_idx = self.helper.BC_zero_index

    def transform(self, u):
        assert u.ndim > 1, 'u must not be flattened here!'
        u_hat = self.helper.transform(u, axes=(-1, -2))
        return u_hat

    def itransform(self, u_hat):
        assert u_hat.ndim > 1, 'u_hat must not be flattened here!'
        u = self.helper.itransform(u_hat, axes=(-2, -1))
        return u

    def compute_derivatives(self, u, skip_transform=False):
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
        assert (
            self.BCs['v_top'] == self.BCs['v_bottom']
        ), 'Initial conditions are only implemented for zero velocity gradient'

        me = self.u_init

        # linear temperature gradient
        for i, n in zip([self.iT, self.iv], ['T', 'v']):
            me[i] = (self.BCs[f'{n}_top'] - self.BCs[f'{n}_bottom']) / 2 * self.Z + (
                self.BCs[f'{n}_top'] + self.BCs[f'{n}_bottom']
            ) / 2.0

        # perturb slightly
        # me[self.iT] += self.xp.random.rand(*me[self.iT].shape) * 1e-3

        # evaluate derivatives
        derivatives = self.compute_derivatives(me)
        for i in [self.iTx, self.iTz, self.iux, self.ivz]:
            me[i] = derivatives[i]

        return me

    def _put_BCs_in_rhs(self, rhs):
        assert rhs.ndim > 1, 'Right hand side must not be flattened here!'

        _rhs_hat = self.helper.transform(rhs, axes=(-1,))

        # _rhs_hat[self.iua, :, -1] = self.BCs['u_top']
        # _rhs_hat[self.iub, :, -1] = self.BCs['u_bottom']
        _rhs_hat[self.iv, :, -1] = self.BCs['v_top']
        _rhs_hat[self.ivz, :, -1] = self.BCs['v_bottom']
        _rhs_hat[self.iT, :, -1] = self.BCs['T_top']
        _rhs_hat[self.iTz, :, -1] = self.BCs['T_bottom']
        _rhs_hat[self.ip, :, -1] = self.BCs['p_top']
        # TODO: I don't think the tau terms are properly distributed!

        return self.helper.itransform(_rhs_hat, axes=(-1,))

    def solve_system(self, rhs, factor, *args, **kwargs):
        sol = self.u_init

        # _rhs = self._put_BCs_in_rhs(rhs)
        _rhs = self.helper.put_BCs_in_rhs(rhs)
        rhs_hat = self.transform(_rhs)

        A = self.M + factor * self.L
        A = self.helper.put_BCs_in_matrix(A)

        sol_hat = self.helper.sparse_lib.linalg.spsolve(A.tocsc(), rhs_hat.flatten()).reshape(sol.shape)

        sol[:] = self.itransform(sol_hat)
        return sol

    def compute_vorticity(self, u):
        u_hat = self.transform(u)
        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx
        vorticity_hat = (Dx * u_hat[self.iv].flatten() + Dz @ u_hat[self.iu].flatten()).reshape(u[self.iu].shape)
        return self.itransform(vorticity_hat)

    def compute_constraint_violation(self, u):
        derivatives = self.compute_derivatives(u)

        violations = {}

        for i in [self.iux, self.ivz, self.iTx, self.iTz]:
            violations[self.index_to_name[i]] = derivatives[i] - u[i]

        violations['divergence'] = u[self.iux] - u[self.ivz]

        return violations

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
        imV = axs[1].pcolormesh(self.X, self.Z, self.compute_vorticity(u).real)

        for i, label in zip([0, 1], [r'$T$', 'vorticity']):
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2e}')
        axs[1].set_xlabel(r'$x$')
        axs[1].set_ylabel(r'$z$')
        fig.colorbar(imT, self.cax[0])
        fig.colorbar(imV, self.cax[1])
