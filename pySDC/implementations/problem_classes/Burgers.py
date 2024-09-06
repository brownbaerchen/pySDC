import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear


class Burgers1D(GenericSpectralLinear):
    """
    See https://en.wikipedia.org/wiki/Burgers'_equation
    Discretization is done with a Chebychov method, which requires a first order derivative formulation.
    Feel free to do a more efficient implementation using an ultraspherical method to avoid the first order business.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, N=64, epsilon=0.1, BCl=1, BCr=-1, f=0, mode='T2U', **kwargs):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        bases = [{'base': 'cheby', 'N': N}]
        components = ['u', 'ux']

        super().__init__(bases=bases, components=components, **kwargs)

        self.x = self.get_grid()[0]

        # prepare matrices
        Dx = self.get_differentiation_matrix(axes=(0,))
        I = self.get_Id()

        T2U = self.get_basis_change_matrix(conv=mode)

        self.Dx = Dx

        # construct linear operator
        L_lhs = {'u': {'ux': -epsilon * (T2U @ Dx)}, 'ux': {'u': -T2U @ Dx, 'ux': T2U @ I}}
        self.setup_L(L_lhs)

        # construct mass matrix
        M_lhs = {'u': {'u': T2U @ I}}
        self.setup_M(M_lhs)

        # boundary conditions
        self.add_BC(component='u', equation='u', axis=0, x=1, v=BCr, kind='Dirichlet')
        self.add_BC(component='u', equation='ux', axis=0, x=-1, v=BCl, kind='Dirichlet')
        self.setup_BCs()

    def u_exact(self, t=0, *args, **kwargs):
        me = self.u_init

        # x = (self.x + 1) / 2
        # g = 4 * (1 + np.exp(-(4 * x + t)/self.epsilon/32))
        # g_x = 4 * np.exp(-(4 * x + t)/self.epsilon/32) * (-4/self.epsilon/32)

        # me[0] = 3./4. - 1./g
        # me[1] = 1/g**2 * g_x

        # return me

        if t == 0:
            me[self.index('u')][:] = ((self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x) * np.cos(
                self.x * np.pi * self.f
            )
            me[self.index('ux')][:] = (self.BCr - self.BCl) / 2 * np.cos(self.x * np.pi * self.f) + (
                (self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x
            ) * self.f * np.pi * -np.sin(self.x * np.pi * self.f)
        elif t == np.inf and self.f == 0 and self.BCl == -self.BCr:
            me[0] = (self.BCl * np.exp((self.BCr - self.BCl) / (2 * self.epsilon) * self.x) + self.BCr) / (
                np.exp((self.BCr - self.BCl) / (2 * self.epsilon) * self.x) + 1
            )
        else:
            raise NotImplementedError

        return me

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iux = self.index('u'), self.index('ux')

        u_hat = self.transform(u)

        Dx_u_hat = self.u_init_forward
        Dx_u_hat[iux] = (self.Dx @ u_hat[iux].flatten()).reshape(u_hat[iu].shape)

        f.impl[iu] = self.epsilon * self.itransform(Dx_u_hat)[iux].real
        f.expl[iu] = -u[iu] * u[iux]
        return f

    def get_fig(self):  # pragma: no cover
        """
        Get a figure suitable to plot the solution of this problem

        Returns
        -------
        self.fig : matplotlib.pyplot.figure.Figure
        """
        import matplotlib.pyplot as plt

        plt.rcParams['figure.constrained_layout.use'] = True
        self.fig, axs = plt.subplots()
        return self.fig

    def plot(self, u, t=None, fig=None, comp='u'):  # pragma: no cover
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
        ax = fig.axes[0]

        ax.plot(self.x, u[self.index(comp)])

        if t is not None:
            fig.suptitle(f't = {t:.2e}')

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')


class Burgers2D(GenericSpectralLinear):
    """
    See documentation of `Burgers1D`.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nx=64, nz=64, epsilon=0.1, fux=2, fuz=1, mode='T2U', **kwargs):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        bases = [
            {'base': 'fft', 'N': nx},
            {'base': 'cheby', 'N': nz},
        ]
        components = ['u', 'v', 'ux', 'uz', 'vx', 'vz']
        super().__init__(bases=bases, components=components, **kwargs)

        self.Z, self.X = self.get_grid()

        # prepare matrices
        Dx = self.get_differentiation_matrix(axes=(0,))
        Dz = self.get_differentiation_matrix(axes=(1,))
        I = self.get_Id()

        T2U = self.get_basis_change_matrix(axes=(1,), conv=mode)

        self.Dx = Dx
        self.Dz = Dz

        # construct linear operator
        L_lhs = {
            'u': {'ux': -epsilon * T2U @ Dx, 'uz': -epsilon * T2U @ Dz},
            'v': {'vx': -epsilon * T2U @ Dx, 'vz': -epsilon * T2U @ Dz},
            'ux': {'u': -T2U @ Dx, 'ux': T2U @ I},
            'uz': {'u': -T2U @ Dz, 'uz': T2U @ I},
            'vx': {'v': -T2U @ Dx, 'vx': T2U @ I},
            'vz': {'v': -T2U @ Dz, 'vz': T2U @ I},
        }
        self.setup_L(L_lhs)

        # construct mass matrix
        M_lhs = {
            'u': {'u': T2U @ I},
            'v': {'v': T2U @ I},
        }
        self.setup_M(M_lhs)

        # boundary conditions
        self.BCtop = 1
        self.BCbottom = -self.BCtop
        self.BCtopu = 0
        self.add_BC(component='v', equation='v', axis=1, v=self.BCtop, x=1, kind='Dirichlet')
        self.add_BC(component='v', equation='vz', axis=1, v=self.BCbottom, x=-1, kind='Dirichlet')
        self.add_BC(component='u', equation='uz', axis=1, v=self.BCtopu, x=1, kind='Dirichlet')
        self.add_BC(component='u', equation='u', axis=1, v=self.BCtopu, x=-1, kind='Dirichlet')
        self.setup_BCs()

    def u_exact(self, t=0, *args, noise_level=0, **kwargs):
        me = self.u_init

        iu, iv, iux, iuz, ivx, ivz = self.index(self.components)
        if t == 0:
            me[iu] = self.xp.cos(self.X * self.fux) * self.xp.sin(self.Z * np.pi * self.fuz) + self.BCtopu
            me[iux] = -self.xp.sin(self.X * self.fux) * self.fux * self.xp.sin(self.Z * np.pi * self.fuz)
            me[iuz] = self.xp.cos(self.X * self.fux) * self.xp.cos(self.Z * np.pi * self.fuz) * np.pi * self.fuz

            me[iv] = (self.BCtop + self.BCbottom) / 2 + (self.BCtop - self.BCbottom) / 2 * self.Z
            me[ivz][:] = (self.BCtop - self.BCbottom) / 2

            # add noise
            rng = self.xp.random.default_rng(seed=99)
            me[iv].real += rng.normal(size=me[iv].shape) * (self.Z - 1) * (self.Z + 1) * noise_level

        else:
            raise NotImplementedError

        return me

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iv, iux, iuz, ivx, ivz = self.index(self.components)

        u_hat = self.transform(u)
        f_hat = self.u_init_forward
        f_hat[iu] = self.epsilon * ((self.Dx @ u_hat[iux].flatten() + self.Dz @ u_hat[iuz].flatten())).reshape(
            u_hat[iux].shape
        )
        f_hat[iv] = self.epsilon * ((self.Dx @ u_hat[ivx].flatten() + self.Dz @ u_hat[ivz].flatten())).reshape(
            u_hat[iux].shape
        )
        f.impl[...] = self.itransform(f_hat).real

        f.expl[iu] = -(u[iu] * u[iux] + u[iv] * u[iuz])
        f.expl[iv] = -(u[iu] * u[ivx] + u[iv] * u[ivz])
        return f

    def compute_vorticity(self, u):
        me = self.u_init_forward

        u_hat = self.transform(u)
        iu, iv = self.index(['u', 'v'])

        me[iu] = (self.Dx @ u_hat[iv].flatten() + self.Dz @ u_hat[iu].flatten()).reshape(u[iu].shape)
        return self.itransform(me)[iu].real

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
        self.fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=((8, 7)))
        self.cax = []
        divider = make_axes_locatable(axs[0])
        self.cax += [divider.append_axes('right', size='3%', pad=0.03)]
        divider2 = make_axes_locatable(axs[1])
        self.cax += [divider2.append_axes('right', size='3%', pad=0.03)]
        divider3 = make_axes_locatable(axs[2])
        self.cax += [divider3.append_axes('right', size='3%', pad=0.03)]
        return self.fig

    def plot(self, u, t=None, fig=None, vmin=None, vmax=None):  # pragma: no cover
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

        iu, iv = self.index(['u', 'v'])

        imU = axs[0].pcolormesh(self.X, self.Z, u[iu].real, vmin=vmin, vmax=vmax)
        imV = axs[1].pcolormesh(self.X, self.Z, u[iv].real, vmin=vmin, vmax=vmax)
        imVort = axs[2].pcolormesh(self.X, self.Z, self.compute_vorticity(u).real)

        for i, label in zip([0, 1, 2], [r'$u$', '$v$', 'vorticity']):
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2e}')
        axs[-1].set_xlabel(r'$x$')
        axs[-1].set_ylabel(r'$z$')
        fig.colorbar(imU, self.cax[0])
        fig.colorbar(imV, self.cax[1])
        fig.colorbar(imVort, self.cax[2])
