import numpy as np
from mpi4py import MPI

from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.core.convergence_controller import ConvergenceController
from pySDC.core.hooks import Hooks
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence
from pySDC.core.problem import WorkCounter


class RayleighBenard3D(GenericSpectralLinear):
    """
    Rayleigh-Benard Convection is a variation of incompressible Navier-Stokes.

    The equations we solve are

        u_x + v_z = 0
        T_t - kappa (T_xx + T_zz) = -uT_x - vT_z
        u_t - nu (u_xx + u_zz) + p_x = -uu_x - vu_z
        v_t - nu (v_xx + v_zz) + p_z - T = -uv_x - vv_z

    with u the horizontal velocity, v the vertical velocity (in z-direction), T the temperature, p the pressure, indices
    denoting derivatives, kappa=(Rayleigh * Prandtl)**(-1/2) and nu = (Rayleigh / Prandtl)**(-1/2). Everything on the left
    hand side, that is the viscous part, the pressure gradient and the buoyancy due to temperature are treated
    implicitly, while the non-linear convection part on the right hand side is integrated explicitly.

    The domain, vertical boundary conditions and pressure gauge are

        Omega = [0, 8) x (-1, 1)
        T(z=+1) = 0
        T(z=-1) = 2
        u(z=+-1) = v(z=+-1) = 0
        integral over p = 0

    The spectral discretization uses FFT horizontally, implying periodic BCs, and an ultraspherical method vertically to
    facilitate the Dirichlet BCs.

    Parameters:
        Prandtl (float): Prandtl number
        Rayleigh (float): Rayleigh number
        nx (int): Horizontal resolution
        nz (int): Vertical resolution
        BCs (dict): Can specify boundary conditions here
        dealiasing (float): Dealiasing factor for evaluating the non-linear part
        comm (mpi4py.Intracomm): Space communicator
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(
        self,
        Prandtl=1,
        Rayleigh=2e6,
        nx=256,
        ny=256,
        nz=64,
        BCs=None,
        dealiasing=1.0,
        comm=None,
        Lx=8,
        Ly=8,
        **kwargs,
    ):
        """
        Constructor. `kwargs` are forwarded to parent class constructor.

        Args:
            Prandtl (float): Prandtl number
            Rayleigh (float): Rayleigh number
            nx (int): Resolution in x-direction
            nz (int): Resolution in z direction
            BCs (dict): Vertical boundary conditions
            dealiasing (float): Dealiasing for evaluating the non-linear part in real space
            comm (mpi4py.Intracomm): Space communicator
            Lx (float): Horizontal length of the domain
        """
        # TODO: documentation
        BCs = {} if BCs is None else BCs
        BCs = {
            'T_top': 0,
            'T_bottom': 2,
            'w_top': 0,
            'w_bottom': 0,
            'v_top': 0,
            'v_bottom': 0,
            'u_top': 0,
            'u_bottom': 0,
            'p_integral': 0,
            **BCs,
        }
        if comm is None:
            try:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
            except ModuleNotFoundError:
                pass
        self._makeAttributeAndRegister(
            'Prandtl',
            'Rayleigh',
            'nx',
            'ny',
            'nz',
            'BCs',
            'dealiasing',
            'comm',
            'Lx',
            'Ly',
            localVars=locals(),
            readOnly=True,
        )

        bases = [
            {'base': 'fft', 'N': nx, 'x0': 0, 'x1': self.Lx},
            {'base': 'fft', 'N': ny, 'x0': 0, 'x1': self.Ly},
            {'base': 'ultraspherical', 'N': nz},
        ]
        components = ['u', 'v', 'w', 'T', 'p']
        super().__init__(bases, components, comm=comm, **kwargs)

        self.X, self.Y, self.Z = self.get_grid()
        self.Kx, self.Ky, self.Kz = self.get_wavenumbers()

        # construct 3D matrices
        Dzz = self.get_differentiation_matrix(axes=(2,), p=2)
        Dz = self.get_differentiation_matrix(axes=(2,))
        Dy = self.get_differentiation_matrix(axes=(1,))
        Dyy = self.get_differentiation_matrix(axes=(1,), p=2)
        Dx = self.get_differentiation_matrix(axes=(0,))
        Dxx = self.get_differentiation_matrix(axes=(0,), p=2)
        Id = self.get_Id()

        S1 = self.get_basis_change_matrix(p_out=0, p_in=1)
        S2 = self.get_basis_change_matrix(p_out=0, p_in=2)

        U01 = self.get_basis_change_matrix(p_in=0, p_out=1)
        U12 = self.get_basis_change_matrix(p_in=1, p_out=2)
        U02 = self.get_basis_change_matrix(p_in=0, p_out=2)

        self.Dx = Dx
        self.Dxx = Dxx
        self.Dy = Dy
        self.Dyy = Dyy
        self.Dz = S1 @ Dz
        self.Dzz = S2 @ Dzz

        # compute rescaled Rayleigh number to extract viscosity and thermal diffusivity
        Ra = Rayleigh / (max([abs(BCs['T_top'] - BCs['T_bottom']), np.finfo(float).eps]) * self.axes[2].L ** 3)
        self.kappa = (Ra * Prandtl) ** (-1 / 2.0)
        self.nu = (Ra / Prandtl) ** (-1 / 2.0)

        # construct operators
        L_lhs = {
            'p': {'u': U01 @ Dx, 'v': U01 @ Dy, 'w': Dz},  # divergence free constraint
            'u': {'p': U02 @ Dx, 'u': -self.nu * (U02 @ (Dxx + Dyy) + Dzz)},
            'v': {'p': U02 @ Dy, 'v': -self.nu * (U02 @ (Dxx + Dyy) + Dzz)},
            'w': {'p': U12 @ Dz, 'w': -self.nu * (U02 @ (Dxx + Dyy) + Dzz), 'T': -U02 @ Id},
            'T': {'T': -self.kappa * (U02 @ (Dxx + Dyy) + Dzz)},
        }
        self.setup_L(L_lhs)

        # mass matrix
        M_lhs = {i: {i: U02 @ Id} for i in ['u', 'v', 'w', 'T']}
        self.setup_M(M_lhs)

        # Prepare going from second (first for divergence free equation) derivative basis back to Chebychov-T
        self.base_change = self._setup_operator({**{comp: {comp: S2} for comp in ['u', 'v', 'w', 'T']}, 'p': {'p': S1}})

        # BCs
        self.add_BC(
            component='p', equation='p', axis=2, v=self.BCs['p_integral'], kind='integral', line=-1, scalar=True
        )
        self.add_BC(component='T', equation='T', axis=2, x=-1, v=self.BCs['T_bottom'], kind='Dirichlet', line=-1)
        self.add_BC(component='T', equation='T', axis=2, x=1, v=self.BCs['T_top'], kind='Dirichlet', line=-2)
        self.add_BC(component='w', equation='w', axis=2, x=1, v=self.BCs['w_top'], kind='Dirichlet', line=-1)
        self.add_BC(component='w', equation='w', axis=2, x=-1, v=self.BCs['w_bottom'], kind='Dirichlet', line=-2)
        self.remove_BC(component='w', equation='w', axis=2, x=-1, kind='Dirichlet', line=-2, scalar=True)
        for comp in ['u', 'v']:
            self.add_BC(
                component=comp, equation=comp, axis=2, v=self.BCs[f'{comp}_top'], x=1, kind='Dirichlet', line=-2
            )
            self.add_BC(
                component=comp,
                equation=comp,
                axis=2,
                v=self.BCs[f'{comp}_bottom'],
                x=-1,
                kind='Dirichlet',
                line=-1,
            )

        # eliminate Nyquist mode if needed
        if nx % 2 == 0:
            Nyquist_mode_index = self.axes[0].get_Nyquist_mode_index()
            for component in self.components:
                self.add_BC(
                    component=component, equation=component, axis=0, kind='Nyquist', line=int(Nyquist_mode_index), v=0
                )
        if ny % 2 == 0:
            Nyquist_mode_index = self.axes[0].get_Nyquist_mode_index()
            for component in self.components:
                self.add_BC(
                    component=component, equation=component, axis=1, kind='Nyquist', line=int(Nyquist_mode_index), v=0
                )
        self.setup_BCs()

        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init

        if self.spectral_space:
            u_hat = u.copy()
        else:
            u_hat = self.transform(u)

        f_impl_hat = self.u_init_forward

        Dz = self.Dz
        Dy = self.Dy
        Dx = self.Dx

        iu, iv, iw, iT, ip = self.index(['u', 'v', 'w', 'T', 'p'])

        # evaluate implicit terms
        if not hasattr(self, '_L_T_base'):
            self._L_T_base = self.base_change @ self.L
        f_impl_hat = -(self._L_T_base @ u_hat.flatten()).reshape(u_hat.shape)

        if self.spectral_space:
            f.impl[:] = f_impl_hat
        else:
            f.impl[:] = self.itransform(f_impl_hat).real

        # -------------------------------------------
        # treat convection explicitly with dealiasing

        # start by computing derivatives
        if not hasattr(self, '_Dx_expanded') or not hasattr(self, '_Dz_expanded'):
            self._Dx_expanded = self._setup_operator(
                {'u': {'u': Dx}, 'v': {'v': Dx}, 'w': {'w': Dx}, 'T': {'T': Dx}, 'p': {}}
            )
            self._Dy_expanded = self._setup_operator(
                {'u': {'u': Dy}, 'v': {'v': Dy}, 'w': {'w': Dy}, 'T': {'T': Dx}, 'p': {}}
            )
            self._Dz_expanded = self._setup_operator(
                {'u': {'u': Dz}, 'v': {'v': Dz}, 'w': {'w': Dz}, 'T': {'T': Dz}, 'p': {}}
            )
        Dx_u_hat = (self._Dx_expanded @ u_hat.flatten()).reshape(u_hat.shape)
        Dy_u_hat = (self._Dy_expanded @ u_hat.flatten()).reshape(u_hat.shape)
        Dz_u_hat = (self._Dz_expanded @ u_hat.flatten()).reshape(u_hat.shape)

        padding = [
            self.dealiasing,
        ] * self.ndim
        Dx_u_pad = self.itransform(Dx_u_hat, padding=padding).real
        Dy_u_pad = self.itransform(Dy_u_hat, padding=padding).real
        Dz_u_pad = self.itransform(Dz_u_hat, padding=padding).real
        u_pad = self.itransform(u_hat, padding=padding).real

        fexpl_pad = self.xp.zeros_like(u_pad)
        fexpl_pad[iu][:] = -(u_pad[iu] * Dx_u_pad[iu] + u_pad[iv] * Dy_u_pad[iu] + u_pad[iw] * Dz_u_pad[iu])
        fexpl_pad[iv][:] = -(u_pad[iu] * Dx_u_pad[iv] + u_pad[iv] * Dy_u_pad[iv] + u_pad[iw] * Dz_u_pad[iv])
        fexpl_pad[iw][:] = -(u_pad[iu] * Dx_u_pad[iw] + u_pad[iv] * Dy_u_pad[iw] + u_pad[iw] * Dz_u_pad[iw])
        fexpl_pad[iT][:] = -(u_pad[iu] * Dx_u_pad[iT] + u_pad[iv] * Dy_u_pad[iT] + u_pad[iw] * Dz_u_pad[iT])

        if self.spectral_space:
            f.expl[:] = self.transform(fexpl_pad, padding=padding)
        else:
            f.expl[:] = self.itransform(self.transform(fexpl_pad, padding=padding)).real

        self.work_counters['rhs']()
        return f

    def u_exact(self, t=0, noise_level=1e-3, seed=99):
        assert t == 0
        assert (
            self.BCs['v_top'] == self.BCs['v_bottom']
        ), 'Initial conditions are only implemented for zero velocity gradient'

        me = self.spectral.u_init
        iu, iw, iT, ip = self.index(['u', 'w', 'T', 'p'])

        # linear temperature gradient
        for comp in ['T', 'u', 'v', 'w']:
            a = (self.BCs[f'{comp}_top'] - self.BCs[f'{comp}_bottom']) / 2
            b = (self.BCs[f'{comp}_top'] + self.BCs[f'{comp}_bottom']) / 2
            me[self.index(comp)] = a * self.Z + b

        # perturb slightly
        rng = self.xp.random.default_rng(seed=seed)

        noise = self.spectral.u_init
        noise[iT] = rng.random(size=me[iT].shape)

        me[iT] += noise[iT].real * noise_level * (self.Z - 1) * (self.Z + 1)

        if self.spectral_space:
            me_hat = self.spectral.u_init_forward
            me_hat[:] = self.transform(me)
            return me_hat
        else:
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
        self.fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=((10, 5)))
        self.cax = []
        divider = make_axes_locatable(axs[0])
        self.cax += [divider.append_axes('right', size='3%', pad=0.03)]
        divider2 = make_axes_locatable(axs[1])
        self.cax += [divider2.append_axes('right', size='3%', pad=0.03)]
        return self.fig

    def plot(self, u, t=None, fig=None, quantity='T'):  # pragma: no cover
        r"""
        Plot the solution.

        Parameters
        ----------
        u : dtype_u
            Solution to be plotted
        t : float
            Time to display at the top of the figure
        fig : matplotlib.pyplot.figure.Figure
            Figure with the same structure as a figure generated by `self.get_fig`. If none is supplied, a new figure will be generated.
        quantity : (str)
            quantity you want to plot

        Returns
        -------
        None
        """
        fig = self.get_fig() if fig is None else fig
        axs = fig.axes

        imV = axs[1].pcolormesh(self.X, self.Z, self.compute_vorticity(u).real)

        if self.spectral_space:
            u = self.itransform(u)

        imT = axs[0].pcolormesh(self.X, self.Z, u[self.index(quantity)].real)

        for i, label in zip([0, 1], [rf'${quantity}$', 'vorticity']):
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2f}')
        axs[1].set_xlabel(r'$x$')
        axs[1].set_ylabel(r'$z$')
        fig.colorbar(imT, self.cax[0])
        fig.colorbar(imV, self.cax[1])

    def compute_vorticity(self, u):
        raise NotImplementedError
        if self.spectral_space:
            u_hat = u.copy()
        else:
            u_hat = self.transform(u)
        Dz = self.Dz
        Dx = self.Dx
        iu, iw = self.index(['u', 'w'])

        vorticity_hat = self.spectral.u_init_forward
        vorticity_hat[0] = (Dx * u_hat[iw].flatten() + Dz @ u_hat[iu].flatten()).reshape(u[iu].shape)
        return self.itransform(vorticity_hat)[0].real
