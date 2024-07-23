import numpy as np

from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class RayleighBenard(GenericSpectralLinear):

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, Pr=1, Ra=1, nx=8, nz=8, cheby_mode='T2T', BCs=None, comm=None):
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

        bases = [{'base': 'fft', 'N': nx}, {'base': 'chebychov', 'N': nz, 'mode': cheby_mode}]
        components = ['u', 'v', 'vz', 'T', 'Tz', 'p']
        super().__init__(bases, components, comm)

        self.indices = {
            'u': 0,
            'v': 1,
            'vz': 2,
            'T': 3,
            'Tz': 4,
            'p': 5,
        }
        self.index_to_name = {v: k for k, v, in self.indices.items()}

        # prepare indexes for the different components. Do this by hand so that the editor can autocomplete...
        self.iu = self.indices['u']  # velocity in x-direction
        self.iv = self.indices['v']  # velocity in z-direction
        self.ivz = self.indices['vz']  # derivative of v wrt z
        self.iT = self.indices['T']  # temperature
        self.iTz = self.indices['Tz']  # derivative of temperature wrt z
        self.ip = self.indices['p']  # pressure

        self.Z, self.X = self.get_grid()

        # construct 2D matrices
        Dx = self.get_differentiation_matrix(axes=(0,))
        Dxx = self.get_differentiation_matrix(axes=(0,), p=2)
        Dz = self.get_differentiation_matrix(axes=(1,))
        I = self.get_Id()
        self.Dx = Dx
        self.Dxx = Dxx
        self.Dz = Dz
        self.U2T = self.get_basis_change_matrix()
        S2D = self.get_integration_matrix(axes=(0, 1))

        # construct operators
        L_lhs = {
            'vz': {'v': -Dz, 'vz': I},
            'Tz': {'T': -Dz, 'Tz': I},
            'p': {'u': -Dx, 'vz': I},  # divergence free constraint
            'u': {'p': Pr * Dx, 'u': -Pr * Dxx},
            'v': {'p': Pr * Dz, 'vz': -Pr * Dz, 'T': -Pr * Ra * I},
            'T': {'T': -Dxx, 'Tz': -Dz},
            # 'p': {'u': -Dx, 'vz': I, 'p': S2D},
        }
        self.setup_L(L_lhs)

        # mass matrix
        M_lhs = {i: {i: I} for i in ['u', 'v', 'T']}
        self.setup_M(M_lhs)

        # TODO: distribute tau terms sensibly
        self.add_BC(component='p', equation='p', axis=1, x=1, v=self.BCs['p_top'])
        self.add_BC(component='v', equation='v', axis=1, x=1, v=self.BCs['v_top'])
        self.add_BC(component='v', equation='vz', axis=1, x=-1, v=self.BCs['v_bottom'])
        self.add_BC(component='T', equation='T', axis=1, x=1, v=self.BCs['T_top'])
        self.add_BC(component='T', equation='Tz', axis=1, x=-1, v=self.BCs['T_bottom'])
        self.setup_BCs()

    def compute_derivatives(self, u, skip_transform=False):
        me_hat = self.u_init

        u_hat = u if skip_transform else self.transform(u)
        shape = u[0].shape
        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx

        for comp1, comp2 in zip(['v', 'T'], ['vz', 'Tz']):
            i = self.index(comp1)
            iD = self.index(comp2)
            me_hat[iD][:] = (Dz @ u_hat[i].flatten()).reshape(shape)

        return me_hat if skip_transform else self.itransform(me_hat)

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init

        u_hat = self.transform(u)
        f_hat = self.f_init

        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx
        Dxx = self.U2T @ self.Dxx

        shape = u[0].shape
        iu, iv, ivz, iT, iTz, ip = self.index(self.components)

        # evaluate implicit terms
        f_hat.impl[iT] = (Dxx @ u_hat[iT].flatten() + Dz @ u_hat[iTz].flatten()).reshape(shape)
        f_hat.impl[iu] = (-Dx @ u_hat[ip].flatten() + Dxx @ u_hat[iu].flatten()).reshape(shape)
        f_hat.impl[iv] = (-Dz @ u_hat[ip].flatten() + Dz @ u_hat[ivz].flatten()).reshape(shape) + self.Ra * u_hat[iT]

        f.impl[:] = self.itransform(f_hat.impl)

        Dx_u_hat = self.u_init
        for i in [iu, iT]:
            Dx_u_hat[i][:] = (Dx @ u_hat[i].flatten()).reshape(shape)
        Dx_u = self.itransform(Dx_u_hat)

        # treat convection explicitly
        f.expl[iu] = -u[iu] * (Dx_u[iu] + u[ivz])
        f.expl[iv] = -u[iv] * (Dx_u[iu] + u[ivz])
        f.expl[iT] = -u[iu] * Dx_u[iT] - u[iv] * u[iTz]

        return f

    def u_exact(self, t=0):
        assert t == 0
        assert (
            self.BCs['v_top'] == self.BCs['v_bottom']
        ), 'Initial conditions are only implemented for zero velocity gradient'

        me = self.u_init

        # linear temperature gradient
        for comp in ['T', 'v']:
            a = (self.BCs[f'{comp}_top'] - self.BCs[f'{comp}_bottom']) / 2
            b = (self.BCs[f'{comp}_top'] + self.BCs[f'{comp}_bottom']) / 2
            me[self.index(comp)] = a * self.Z + b
            me[self.index(f'{comp}z')] = a

        # perturb slightly
        # me[self.iT] += self.xp.random.rand(*me[self.iT].shape) * 1e-3

        # evaluate derivatives
        # derivatives = self.compute_derivatives(me)
        # for comp in ['Tz', 'vz']:
        #     i = self.index(comp)
        #     me[i] = derivatives[i]

        return me

    def compute_vorticity(self, u):
        u_hat = self.transform(u)
        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx
        iu = self.index('u')

        vorticity_hat = self.u_init
        vorticity_hat[0] = (Dx * u_hat[self.iv].flatten() + Dz @ u_hat[iu].flatten()).reshape(u[iu].shape)
        return self.itransform(vorticity_hat)[0]

    # def compute_constraint_violation(self, u):
    #     derivatives = self.compute_derivatives(u)

    #     violations = {}

    #     for i in [self.ivz, self.iTz]:
    #         violations[self.index_to_name[i]] = derivatives[i] - u[i]

    #     violations['divergence'] = u[self.iux] - u[self.ivz]

    #     return violations

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
