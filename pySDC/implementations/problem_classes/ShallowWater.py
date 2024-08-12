from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
import numpy as np


class ShallowWaterLinearized(GenericSpectralLinear):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nx=2**6, ny=2**6, f=0, k=1, g=1, H=1e2, **kwargs):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        # bases = [{'base': 'fft', 'N': nx, 'x0': -1, 'x1': 1}, {'base': 'fft', 'N': ny, 'x0': -1, 'x1': 1}]
        bases = [{'base': 'fft', 'N': nx, 'x0': 0, 'x1': 2 * np.pi}, {'base': 'fft', 'N': ny, 'x0': 0, 'x1': 2 * np.pi}]
        components = ['h', 'u', 'v']

        super().__init__(bases, components, **kwargs)

        Dx = self.get_differentiation_matrix(axes=(-2,))
        Dy = self.get_differentiation_matrix(axes=(-1,))
        Id = self.get_Id()
        self.U2T = self.get_basis_change_matrix()
        self.Dx = Dx
        self.Dy = Dy

        L_lhs = {
            'h': {'u': H * Dx, 'v': H * Dy},
            'u': {'v': -f * Id, 'h': g * Dx, 'u': k * Id},
            'v': {'u': f * Id, 'h': g * Dy, 'v': k * Id},
        }
        self.setup_L(L_lhs)

        M_lhs = {comp: {comp: Id} for comp in ['h', 'u', 'v']}
        self.setup_M(M_lhs)

        # for comp in components:
        #     self.add_BC(component=comp, equation=comp, axis=1, x=-1, v=0, kind='Dirichlet')
        self.setup_BCs()

        self.Y, self.X = self.get_grid()

    def eval_f(self, u, *args, **kwargs):
        ih, iu, iv = self.index(self.components)
        f = self.f_init

        f_hat = self.u_init_forward

        u_hat = self.transform(u)
        f_hat[ih] = -self.H * (self.U2T @ (self.Dx @ u_hat[iu].flatten() + self.Dy @ u_hat[iv].flatten())).reshape(
            f[ih].shape
        )
        f_hat[iu] = (
            self.f * u_hat[iv]
            - self.g * (self.U2T @ self.Dx @ u_hat[ih].flatten()).reshape(f_hat[iu].shape)
            - self.k * u_hat[iu]
        )
        f_hat[iv] = (
            -self.f * u_hat[iu]
            - self.g * (self.U2T @ self.Dy @ u_hat[ih].flatten()).reshape(f_hat[iu].shape)
            - self.k * u_hat[iv]
        )

        f[:] = self.itransform(f_hat).real
        return f

    def u_exact(self, t=0):

        assert t == 0 or (self.k == 0 and self.f == 0)

        ih, iu, iv = self.index(self.components)
        xp = self.xp

        me = self.u_init

        gh = np.sqrt(self.H * self.g)
        Lx, Ly = (me.L for me in self.axes)

        amplitudes_x = [1, 2]
        freq_x = [4 * np.pi / Lx, 6 * np.pi / Lx]

        amplitudes_y = [0.1, 1.7]
        freq_y = [2 * np.pi / Ly, 4 * np.pi / Ly]

        for f, A in zip(freq_x, amplitudes_x):
            wave = A * xp.sin((self.X - gh * t) * f)
            me[ih] += wave
            me[iu] += np.sqrt(self.g / self.H) * wave

        for f, A in zip(freq_y, amplitudes_y):
            wave = A * xp.sin((self.Y - gh * t) * f)
            me[ih] += wave
            me[iv] += np.sqrt(self.g / self.H) * wave

        return me

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
        self.fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=((7, 6)))
        divider = make_axes_locatable(axs)
        self.cax = [divider.append_axes('right', size='3%', pad=0.03)]
        return self.fig

    def plot(self, u, t=None, fig=None, vmin=None, vmax=None, comp='h'):  # pragma: no cover
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

        index = self.index(comp)

        im = axs[0].pcolormesh(self.X, self.Y, u[index].real, vmin=vmin, vmax=vmax)

        axs[0].set_aspect(1)
        axs[0].set_title(comp)

        if t is not None:
            fig.suptitle(f't = {t:.2e}')
        axs[-1].set_xlabel(r'$x$')
        axs[-1].set_ylabel(r'$y$')
        fig.colorbar(im, self.cax[0])
