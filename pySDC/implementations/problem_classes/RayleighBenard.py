import numpy as np
from mpi4py import MPI

from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.core.convergence_controller import ConvergenceController
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence


class RayleighBenard(GenericSpectralLinear):
    dtype_u = mesh
    dtype_f = imex_mesh

    def compute_vorticity(self, u):
        u_hat = self.transform(u)
        Dz = self.Dz
        Dx = self.Dx
        iu, iv = self.index(['u', 'v'])

        vorticity_hat = self.u_init_forward
        vorticity_hat[0] = (Dx * u_hat[iv].flatten() + Dz @ u_hat[iu].flatten()).reshape(u[iu].shape)
        return self.itransform(vorticity_hat)[0].real

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

        imT = axs[0].pcolormesh(self.X, self.Z, u[self.index(quantity)].real)
        imV = axs[1].pcolormesh(self.X, self.Z, self.compute_vorticity(u).real)

        for i, label in zip([0, 1], [rf'${quantity}$', 'vorticity']):
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2f}')
        axs[1].set_xlabel(r'$x$')
        axs[1].set_ylabel(r'$z$')
        fig.colorbar(imT, self.cax[0])
        fig.colorbar(imV, self.cax[1])


class RayleighBenardUltraspherical(RayleighBenard):

    def __init__(
        self,
        Prandl=1,
        Rayleigh=2e6,
        nx=257,
        nz=64,
        BCs=None,
        dealiasing=3 / 2,
        comm=None,
        debug=False,
        **kwargs,
    ):
        BCs = {} if BCs is None else BCs
        BCs = {
            'T_top': 0,
            'T_bottom': 2,
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
            'Prandl',
            'Rayleigh',
            'nx',
            'nz',
            'BCs',
            'dealiasing',
            'comm',
            'debug',
            localVars=locals(),
            readOnly=True,
        )

        bases = [{'base': 'fft', 'N': nx, 'x0': 0, 'x1': 8}, {'base': 'ultraspherical', 'N': nz}]
        components = ['u', 'v', 'T', 'p']
        super().__init__(bases, components, comm=comm, **kwargs)

        self.Z, self.X = self.get_grid()
        self.Kz, self.Kx = self.get_wavenumbers()

        # construct 2D matrices
        Dzz = self.get_differentiation_matrix(axes=(1,), p=2)
        Dz = self.get_differentiation_matrix(axes=(1,))
        Dx = self.get_differentiation_matrix(axes=(0,))
        Dxx = self.get_differentiation_matrix(axes=(0,), p=2)
        Id = self.get_Id()

        S1 = self.get_basis_change_matrix(p_out=0, p_in=1)
        S2 = self.get_basis_change_matrix(p_out=0, p_in=2)

        U01 = self.get_basis_change_matrix(p_in=0, p_out=1)
        U12 = self.get_basis_change_matrix(p_in=1, p_out=2)
        U02 = self.get_basis_change_matrix(p_in=0, p_out=2)

        # self.FZ = self.get_filter_matrix(axis=1, kmax=nz-8)

        self.Dx = Dx
        self.Dxx = Dxx
        self.Dz = S1 @ Dz
        self.Dzz = S2 @ Dzz

        kappa = (Rayleigh * Prandl) ** (-1 / 2.0)
        nu = (Rayleigh / Prandl) ** (-1 / 2.0)

        # construct operators
        L_lhs = {
            'p': {'u': U01 @ Dx, 'v': Dz},  # divergence free constraint
            'u': {'p': U02 @ Dx, 'u': -nu * (U02 @ Dxx + Dzz)},
            'v': {'p': U12 @ Dz, 'v': -nu * (U02 @ Dxx + Dzz), 'T': -U02 @ Id},
            'T': {'T': -kappa * (U02 @ Dxx + Dzz)},
        }
        self.setup_L(L_lhs)

        # mass matrix
        M_lhs = {i: {i: U02 @ Id} for i in ['u', 'v', 'T']}
        self.setup_M(M_lhs)

        self.base_change = self._setup_operator({**{comp: {comp: S2} for comp in ['u', 'v', 'T']}, 'p': {'p': S1}})

        self.add_BC(
            component='p', equation='p', axis=1, v=self.BCs['p_integral'], kind='integral', line=-1, scalar=True
        )
        self.add_BC(component='T', equation='T', axis=1, x=-1, v=self.BCs['T_bottom'], kind='Dirichlet', line=-1)
        self.add_BC(component='T', equation='T', axis=1, x=1, v=self.BCs['T_top'], kind='Dirichlet', line=-2)
        self.add_BC(component='v', equation='v', axis=1, x=1, v=self.BCs['v_bottom'], kind='Dirichlet', line=-1)
        self.add_BC(component='v', equation='v', axis=1, x=-1, v=self.BCs['v_bottom'], kind='Dirichlet', line=-2)
        self.remove_BC(component='v', equation='v', axis=1, x=-1, kind='Dirichlet', line=-2, scalar=True)
        self.add_BC(component='u', equation='u', axis=1, v=self.BCs['u_top'], x=1, kind='Dirichlet', line=-2)
        self.add_BC(
            component='u',
            equation='u',
            axis=1,
            v=self.BCs['u_bottom'],
            x=-1,
            kind='Dirichlet',
            line=-1,
        )
        self.setup_BCs()

        if nx % 2 == 0:
            self.logger.warning(
                f'The resolution is x-direction is even at {nx}. Keep in mind that the Nyquist mode is only partially resolved in this case. Consider changing the solution by one.'
            )

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init

        u_hat = self.transform(u)
        f_impl_hat = self.u_init_forward

        Dz = self.Dz
        Dzz = self.Dzz
        Dx = self.Dx
        Dxx = self.Dxx

        shape = u[0].shape
        iu, iv, iT, ip = self.index(['u', 'v', 'T', 'p'])

        kappa = (self.Rayleigh * self.Prandl) ** (-1 / 2)
        nu = (self.Rayleigh / self.Prandl) ** (-1 / 2)

        # evaluate implicit terms
        f_impl_hat = -(self.base_change @ self.L @ u_hat.flatten()).reshape(u_hat.shape)
        f.impl[:] = self.itransform(f_impl_hat).real

        # print(abs(f.impl[ip]))

        # treat convection explicitly with dealiasing
        Dx_u_hat = self.u_init_forward
        for i in [iu, iv, iT]:
            Dx_u_hat[i][:] = (Dx @ u_hat[i].flatten()).reshape(Dx_u_hat[i].shape)
        Dz_u_hat = self.u_init_forward
        for i in [iu, iv, iT]:
            Dz_u_hat[i][:] = (Dz @ u_hat[i].flatten()).reshape(Dz_u_hat[i].shape)

        padding = [self.dealiasing, self.dealiasing]
        Dx_u_pad = self.itransform(Dx_u_hat, padding=padding).real
        Dz_u_pad = self.itransform(Dz_u_hat, padding=padding).real
        u_pad = self.itransform(u_hat, padding=padding).real

        fexpl_pad = self.xp.zeros_like(u_pad)
        fexpl_pad[iu][:] = -(u_pad[iu] * Dx_u_pad[iu] + u_pad[iv] * Dz_u_pad[iu])
        fexpl_pad[iv][:] = -(u_pad[iu] * Dx_u_pad[iv] + u_pad[iv] * Dz_u_pad[iv])
        fexpl_pad[iT][:] = -(u_pad[iu] * Dx_u_pad[iT] + u_pad[iv] * Dz_u_pad[iT])

        f.expl[:] = self.itransform(self.transform(fexpl_pad, padding=padding)).real

        return f

    def u_exact(self, t=0, noise_level=1e-3, seed=99):
        assert t == 0
        assert (
            self.BCs['v_top'] == self.BCs['v_bottom']
        ), 'Initial conditions are only implemented for zero velocity gradient'

        me = self.u_init
        iu, iv, iT, ip = self.index(['u', 'v', 'T', 'p'])

        # linear temperature gradient
        for comp in ['T', 'v', 'u']:
            a = (self.BCs[f'{comp}_top'] - self.BCs[f'{comp}_bottom']) / 2
            b = (self.BCs[f'{comp}_top'] + self.BCs[f'{comp}_bottom']) / 2
            me[self.index(comp)] = a * self.Z + b

        # perturb slightly
        rng = self.xp.random.default_rng(seed=seed)

        noise = self.u_init
        noise[iT] = rng.random(size=me[iT].shape)

        me[iT] += noise[iT].real * noise_level * (self.Z - 1) * (self.Z + 1)

        return me


class RayleighBenardChebychov(GenericSpectralLinear):

    def __init__(
        self,
        Prandl=1,
        Rayleigh=2e6,
        nx=256,
        nz=64,
        cheby_mode='T2U',
        BCs=None,
        dealiasing=3 / 2,
        comm=None,
        debug=False,
        **kwargs,
    ):
        BCs = {} if BCs is None else BCs
        BCs = {
            'T_top': 0,
            'T_bottom': 2,
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
            'Prandl',
            'Rayleigh',
            'nx',
            'nz',
            'cheby_mode',
            'BCs',
            'dealiasing',
            'comm',
            'debug',
            localVars=locals(),
            readOnly=True,
        )

        bases = [{'base': 'fft', 'N': nx, 'x0': 0, 'x1': 8}, {'base': 'chebychov', 'N': nz, 'mode': cheby_mode}]
        components = ['u', 'v', 'vz', 'T', 'Tz', 'p', 'uz']
        super().__init__(bases, components, comm=comm, **kwargs)

        self.indices = {
            'u': 0,
            'v': 1,
            'vz': 2,
            'T': 3,
            'Tz': 4,
            'p': 5,
        }
        self.index_to_name = {v: k for k, v, in self.indices.items()}

        self.Z, self.X = self.get_grid()
        self.Kz, self.Kx = self.get_wavenumbers()

        # construct 2D matrices
        Dx = self.get_differentiation_matrix(axes=(0,))
        Dxx = self.get_differentiation_matrix(axes=(0,), p=2)
        Dz = self.get_differentiation_matrix(axes=(1,))
        Id = self.get_Id()
        self.Dx = Dx
        self.Dxx = Dxx
        self.Dz = Dz
        self.U2T = self.get_basis_change_matrix()

        kappa = (Rayleigh * Prandl) ** (-1 / 2.0)
        nu = (Rayleigh / Prandl) ** (-1 / 2.0)

        # construct operators
        L_lhs = {
            'vz': {'v': -Dz, 'vz': Id},  # algebraic constraint for first derivative
            'uz': {'u': -Dz, 'uz': Id},  # algebraic constraint for first derivative
            'Tz': {'T': -Dz, 'Tz': Id},  # algebraic constraint for first derivative
            'p': {'u': Dx, 'vz': Id},  # divergence free constraint
            'u': {'p': Dx, 'u': -nu * Dxx, 'uz': -nu * Dz},
            'v': {'p': Dz, 'v': -nu * Dxx, 'vz': -nu * Dz, 'T': -Id},
            'T': {'T': -kappa * Dxx, 'Tz': -kappa * Dz},
        }
        self.setup_L(L_lhs)

        # mass matrix
        M_lhs = {i: {i: Id} for i in ['u', 'v', 'T']}
        self.setup_M(M_lhs)

        self.add_BC(
            component='p',
            equation='p',
            axis=0,
            v=self.BCs['p_integral'],
            kind='integral',
            line=0,
            scalar=True,
        )
        # self.add_BC(
        #     component='p', equation='p', axis=1, v=self.BCs['p_integral'], kind='integral', line=-1, scalar=True
        # )
        self.add_BC(component='T', equation='T', axis=1, x=-1, v=self.BCs['T_bottom'], kind='Dirichlet')
        self.add_BC(component='T', equation='Tz', axis=1, x=1, v=self.BCs['T_top'], kind='Dirichlet', line=-1)
        self.add_BC(component='v', equation='v', axis=1, x=-1, v=self.BCs['v_bottom'], kind='Dirichlet')
        self.add_BC(component='v', equation='vz', axis=1, x=1, v=self.BCs['v_top'], kind='Dirichlet', line=-1)
        self.add_BC(component='u', equation='uz', axis=1, v=self.BCs['u_top'], x=1, kind='Dirichlet')
        self.add_BC(
            component='u',
            equation='u',
            axis=1,
            v=self.BCs['u_bottom'],
            x=-1,
            kind='Dirichlet',
            line=-1,
        )
        self.setup_BCs()

    def compute_z_derivatives(self, u):
        me_hat = self.u_init_forward

        u_hat = self.transform(u)
        shape = u[0].shape
        Dz = self.U2T @ self.Dz

        for comp in ['T', 'v', 'u']:
            i = self.index(comp)
            iD = self.index(f'{comp}z')
            me_hat[iD][:] = (Dz @ u_hat[i].flatten()).reshape(shape)

        return self.itransform(me_hat).real

    def eval_f(self, u, *args, compute_violations=True, **kwargs):
        f = self.f_init

        u_hat = self.transform(u)
        f_impl_hat = self.u_init_forward

        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx
        Dxx = self.U2T @ self.Dxx

        shape = u[0].shape
        iu, iv, ivz, iT, iTz, ip, iuz = self.index(self.components)

        if compute_violations:
            f_impl_hat[...] = -(self.base_change @ self.L @ u_hat.flatten()).reshape(u_hat.shape)

            # if compute_violations:
            #     f_impl_hat[iuz][:] = (Dz @ u_hat[iu].flatten()).reshape(shape) - u_hat[iuz]
            #     f_impl_hat[ivz][:] = (Dz @ u_hat[iv].flatten()).reshape(shape) - u_hat[ivz]
            #     f_impl_hat[iTz][:] = (Dz @ u_hat[iT].flatten()).reshape(shape) - u_hat[iTz]
            #     f_impl_hat[ip][:] = (Dz @ u_hat[iv].flatten() + Dx @ u_hat[iu].flatten()).reshape(shape)
        else:
            kappa = (self.Rayleigh * self.Prandl) ** (-1 / 2)
            nu = (self.Rayleigh / self.Prandl) ** (-1 / 2)

            # evaluate implicit terms
            f_impl_hat[iT][:] = kappa * (Dxx @ u_hat[iT].flatten() + Dz @ u_hat[iTz].flatten()).reshape(shape)
            f_impl_hat[iu][:] = (
                -Dx @ u_hat[ip].flatten() + nu * (Dxx @ u_hat[iu].flatten() + Dz @ u_hat[iuz].flatten())
            ).reshape(shape)
            f_impl_hat[iv][:] = (
                -Dz @ u_hat[ip].flatten() + nu * (Dxx @ u_hat[iv].flatten() + Dz @ u_hat[ivz].flatten())
            ).reshape(shape) + u_hat[iT]

        f.impl[:] = self.itransform(f_impl_hat).real

        # treat convection explicitly with dealiasing
        Dx_u_hat = self.u_init_forward
        for i in [iu, iv, iT]:
            Dx_u_hat[i][:] = (Dx @ u_hat[i].flatten()).reshape(Dx_u_hat[i].shape)

        padding = [self.dealiasing, self.dealiasing]
        Dx_u_pad = self.itransform(Dx_u_hat, padding=padding).real
        u_pad = self.itransform(u_hat, padding=padding).real

        fexpl_pad = self.xp.zeros_like(u_pad)
        fexpl_pad[iu][:] = -(u_pad[iu] * Dx_u_pad[iu] + u_pad[iv] * u_pad[iuz])
        fexpl_pad[iv][:] = -(u_pad[iu] * Dx_u_pad[iv] + u_pad[iv] * u_pad[ivz])
        fexpl_pad[iT][:] = -(u_pad[iu] * Dx_u_pad[iT] + u_pad[iv] * u_pad[iTz])

        f.expl[:] = self.itransform(self.transform(fexpl_pad, padding=padding)).real

        return f

    def u_exact(self, t=0, noise_level=1e-3, seed=99, kxmax=None, kzmax=None, raiseExceptions=False):
        assert t == 0
        assert (
            self.BCs['v_top'] == self.BCs['v_bottom']
        ), 'Initial conditions are only implemented for zero velocity gradient'

        me = self.u_init
        iu, iv, ivz, iT, iTz, ip, iuz = self.index(self.components)

        # linear temperature gradient
        for comp in ['T', 'v', 'u']:
            a = (self.BCs[f'{comp}_top'] - self.BCs[f'{comp}_bottom']) / 2
            b = (self.BCs[f'{comp}_top'] + self.BCs[f'{comp}_bottom']) / 2
            me[self.index(comp)] = a * self.Z + b
            me[self.index(f'{comp}z')] = a

        # perturb slightly
        rng = self.xp.random.default_rng(seed=seed)

        noise = self.u_init
        noise[iT] = rng.random(size=me[iT].shape)

        Kz, Kx = self.get_wavenumbers()
        kzmax = self.nz - 3 if kzmax is None else kzmax
        kxmax = self.nx // 2 if kxmax is None else kxmax
        noise_hat = self.u_init_forward
        noise_hat[:] = rng.random(size=noise_hat[iT].shape)
        noise_hat[iT, np.abs(Kx) > kxmax] *= 0
        noise_hat[iT, Kz > kzmax] *= 0
        noise = self.itransform(noise_hat).real

        me[iT] += noise[iT].real * noise_level * (self.Z - 1) * (self.Z + 1)

        # enforce boundary conditions in spite of noise
        me_hat = self.transform(me, axes=(-1,))
        bc_top = self.spectral.axes[1].get_BC(x=1, kind='Dirichlet')
        bc_bottom = self.spectral.axes[1].get_BC(x=-1, kind='Dirichlet')

        if noise_level > 0:
            rhs = self.xp.empty(shape=(2, me_hat.shape[1]), dtype=complex)
            rhs[0] = (
                self.BCs["T_top"]
                - self.xp.sum(bc_top[: kzmax - 2] * me_hat[iT, :, : kzmax - 2], axis=1)
                - self.xp.sum(bc_top[kzmax:] * me_hat[iT, :, kzmax:], axis=1)
            )
            rhs[1] = (
                self.BCs["T_bottom"]
                - self.xp.sum(bc_bottom[: kzmax - 2] * me_hat[iT, :, : kzmax - 2], axis=1)
                - self.xp.sum(bc_bottom[kzmax:] * me_hat[iT, :, kzmax:], axis=1)
            )

            A = self.xp.array([bc_top[kzmax - 2 : kzmax], bc_bottom[kzmax - 2 : kzmax]], complex)
            me_hat[iT, :, kzmax - 2 : kzmax] = self.xp.linalg.solve(A, rhs).T

            me[...] = self.itransform(me_hat, axes=(-1,)).real

        u_hat = self.transform(me)
        # S = self.get_integration_matrix(axes=(1,))
        # # u_hat[ip] = (self.U2T @ S @ u_hat[iT].flatten()).reshape(u_hat[ip].shape)

        u_hat[iTz] = (self.U2T @ self.Dz @ u_hat[iT].flatten()).reshape(u_hat[iTz].shape)
        u_hat[ivz] = (self.U2T @ self.Dz @ u_hat[iv].flatten()).reshape(u_hat[ivz].shape)

        me[...] = self.itransform(u_hat).real
        # # me[ip] += -1.0 / 12.0 * self.BCs['T_top'] + 1 / 12.0 * self.BCs['T_bottom'] + self.BCs['p_integral'] / 2.0

        violations = self.compute_constraint_violation(me)
        for k, v in violations.items():
            if raiseExceptions:
                assert self.xp.allclose(v, 0), f'Initial conditions violate constraint in {k}!'
            elif not self.xp.allclose(v, 0):
                print(f'Initial conditions violate constraint in {k}!')

        violations = self.computeBCviolations(me)
        msg = ''
        for key, value in violations.items():
            if value > 1e-10:
                msg += f' {key}: {value:.2e}'
        if raiseExceptions:
            assert msg == '', f'BC violation in initial conditions: {msg}'
        elif msg != '':
            self.logger.warning(f'BC violation in initial conditions:{msg}')

        return me

    def solve_system(self, *args, **kwargs):
        sol = super().solve_system(*args, **kwargs)

        if self.debug:
            violations = {
                key: self.xp.max(self.xp.abs(value)) for key, value in self.compute_constraint_violation(sol).items()
            }
            msg = ''
            for key, value in violations.items():
                if value > 1e-10:
                    msg += f' {key}: {value:.2e}'
            if msg != '':
                self.logger.warning(f'Constraint violation:{msg}')

            violations = self.computeBCviolations(sol)
            msg = ''
            for key, value in violations.items():
                if value > 1e-10:
                    msg += f' {key}: {value:.2e}'
            if msg != '':
                self.logger.warning(f'BC violation:{msg}')

        return sol

    def compute_constraint_violation(self, u):
        iu, iv, ivz, iT, iTz, ip, iuz = self.index(self.components)
        idxu, idxv, idzu, idzv, idzT = 0, 1, 2, 3, 4

        derivatives_hat = self.u_init_forward

        u_hat = self.transform(u)
        derivatives_hat[idxu] = (self.U2T @ self.Dx @ u_hat[iu].flatten()).reshape(derivatives_hat[idxu].shape)
        derivatives_hat[idxv] = (self.U2T @ self.Dx @ u_hat[iv].flatten()).reshape(derivatives_hat[idxu].shape)
        derivatives_hat[idzu] = (self.U2T @ self.Dz @ u_hat[iu].flatten()).reshape(derivatives_hat[idxu].shape)
        derivatives_hat[idzv] = (self.U2T @ self.Dz @ u_hat[iv].flatten()).reshape(derivatives_hat[idxu].shape)
        derivatives_hat[idzT] = (self.U2T @ self.Dz @ u_hat[iT].flatten()).reshape(derivatives_hat[idxu].shape)
        derivatives = self.itransform(derivatives_hat).real

        violations = {}

        violations['Tz'] = derivatives[idzT] - u[iTz]  # / derivatives[idzT]
        violations['vz'] = derivatives[idzv] - u[ivz]  # / derivatives[idzv]
        violations['uz'] = derivatives[idzu] - u[iuz]  # / derivatives[idzu]

        violations['divergence'] = derivatives[idxu] + derivatives[idzv]

        return violations

    def computeBCviolations(self, u):
        xp = self.xp
        iu, iv, ivz, iT, iTz, ip, iuz = self.index(self.components)

        BC_top = self.axes[-1].get_BC(kind='Dirichlet', x=1)
        BC_bot = self.axes[-1].get_BC(kind='Dirichlet', x=-1)
        BC_int = self.axes[-1].get_BC(kind='integral')

        u_hat = self.transform(u, axes=(-1,))

        violations = {}
        for q in ['T', 'v', 'u']:
            top = u_hat[self.index(q)] @ BC_top
            expect_top = self.BCs[f'{q}_top']
            if not np.allclose(top, expect_top):
                violations[f'{q}_top'] = xp.max(xp.abs(top - expect_top))

            bottom = u_hat[self.index(q)] @ BC_bot
            expect_bottom = self.BCs[f'{q}_bottom']
            if not np.allclose(top, expect_top):
                violations[f'{q}_bottom'] = xp.max(xp.abs(bottom - expect_bottom))

        if not np.allclose(u_hat[self.index('p')] @ BC_int, self.BCs['p_integral']):
            violations['p_integral'] = xp.max(xp.abs(u_hat[self.index('p')] @ BC_int - self.BCs['p_integral']))
        return violations

    def check_refinement_needed(self, u_hat, tol=1e-7, nx_max=np.inf, nz_max=np.inf):
        """
        The derivative is an approximation to the derivative of the exact solution, which may not be the derivative of the numerical solution if the resolution is too low. We check if the derivative has energy in the highest mode, which is an indication of this.
        """
        need_more = False

        if self.nx >= nx_max or self.nz >= nz_max:
            return False

        for i in self.index(['uz', 'vz', 'Tz']):
            need_more = need_more or not self.xp.allclose(u_hat[i][:, -1], 0, atol=tol)

        if self.comm:
            need_more = self.comm.allreduce(need_more, op=MPI.BOR)

        return need_more

    def refine_resolution(self, u_hat, add_modes=2):
        xp = self.xp
        nz_new = self.nz + add_modes
        nx_new = self.nx + add_modes * self.nx // self.nz

        new_params = {**self.params, 'nx': nx_new, 'nz': nz_new}
        P_new = type(self)(**new_params)

        kx = xp.fft.fftfreq(nx_new, 1 / nx_new)[P_new.local_slice[0]]
        mask_x = xp.abs(kx) <= self.nx // 2
        if self.nx % 2 == 0:
            mask_x = xp.logical_and(mask_x, kx != self.nx // 2)
        slices = [slice(0, u_hat.shape[0]), mask_x, slice(0, self.nz)]

        me = P_new.u_init_forward
        me[(*slices,)] = u_hat[...] * nx_new / self.nx
        self.logger.debug(f'Refined spatial resolution by {add_modes} to nx={P_new.nx} and nz={P_new.nz}')

        return me, P_new

    def refine_resolution2(self, u_hat, factor=3 / 2):
        padding = [
            factor,
        ] * 2
        u = self.itransform(u_hat, padding=padding).real

        nz_new = u.shape[2]
        if self.comm:
            nx_new = self.comm.allreduce(u.shape[1], op=MPI.SUM)
        else:
            nx_new = u.shape[1]

        new_params = {**self.params, 'nx': nx_new, 'nz': nz_new, 'useGPU': self.useGPU}
        P_new = type(self)(**new_params)

        me = P_new.u_init
        me[...] = u[...]
        self.logger.debug(f'Refined spatial resolution by {factor} to nx={P_new.nx} and nz={P_new.nz}')

        return me, P_new

    def check_derefinement_ok(self, u_hat, factor=3 / 2, tol=1e-7, nx_min=0, nz_min=0):
        xp = self.xp

        nx_new = int(np.ceil(self.nx / factor))
        nz_new = int(np.ceil(self.nz / factor))

        if nx_new < nx_min or nz_new < nz_min:
            return False

        kx = xp.fft.fftfreq(self.nx, 1 / self.nx)[self.local_slice[0]]
        mask_x = xp.abs(kx) > nx_new // 2
        if not nx_new % 2:
            mask_x = xp.logical_and(mask_x, kx == -nx_new // 2)

        slices = [slice(0, u_hat.shape[0]), mask_x, slice(0, u_hat.shape[2])]
        less_is_fine_x = xp.allclose(u_hat[(*slices,)], 0, atol=tol)

        slices = [slice(0, u_hat.shape[0]), slice(0, u_hat.shape[1]), slice(nz_new, u_hat.shape[2])]
        less_is_fine_z = xp.allclose(u_hat[(*slices,)], 0, atol=tol)

        less_is_fine = less_is_fine_x and less_is_fine_z

        if self.comm:
            less_is_fine = self.comm.allreduce(less_is_fine, op=MPI.BAND)
        return less_is_fine

    def derefine_resolution2(self, u, factor=3 / 2):
        padding = [
            factor,
        ] * 2

        nx_new = int(np.ceil(self.nx / factor))
        nz_new = int(np.ceil(self.nz / factor))

        new_params = {**self.params, 'nx': nx_new, 'nz': nz_new}
        P_new = type(self)(**new_params)

        u_hat = P_new.transform(u, padding=padding).real

        me = P_new.u_init_forward
        me[...] = u_hat[...]
        self.logger.debug(f'Derefined spatial resolution by {factor} to nx={P_new.nx} and nz={P_new.nz}')

        return me, P_new

    def derefine_resolution(self, u_hat, remove_modes=4):
        min_res = 1
        if self.comm:
            min_res = self.comm.size

        xp = self.xp

        nz_new = max([min_res, int(self.nz - remove_modes)])
        nx_new = max([min_res, int(self.nx - remove_modes * self.nx // self.nz)])

        kx = xp.fft.fftfreq(self.nx, 1 / self.nx)
        mask_x = xp.abs(kx) <= nx_new // 2
        if nx_new % 2 == 0:
            mask_x = xp.logical_and(mask_x, kx != nx_new // 2)
        slices = [slice(0, u_hat.shape[0]), mask_x, slice(0, nz_new)]

        new_params = {**self.params, 'nx': nx_new, 'nz': nz_new}
        P_new = type(self)(**new_params)

        me = P_new.u_init_forward
        me[...] = u_hat[(*slices,)] * nx_new / self.nx

        self.logger.debug(f'Derefined spatial resolution by {remove_modes} modes to nx={P_new.nx} and nz={P_new.nz}')
        return me, P_new


class CFLLimit(ConvergenceController):
    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here.

        Default parameters are:
         - control_order (int): The order relative to other convergence controllers
         - dt_max (float): maximal step size
         - dt_min (float): minimal step size

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": -50,
            "dt_max": np.inf,
            "dt_min": 0,
            "cfl": 0.4,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    @staticmethod
    def compute_max_step_size(P, u):
        grid_spacing_x = P.X[1, 0] - P.X[0, 0]
        grid_spacing_z = P.xp.append(P.Z[0, :-1] - P.Z[0, 1:], P.Z[0, -1] - P.axes[1].x0)

        iu, iv = P.index(['u', 'v'])

        max_step_size_x = P.xp.min(grid_spacing_x / P.xp.abs(u[iu]))
        max_step_size_z = P.xp.min(grid_spacing_z / P.xp.abs(u[iv]))
        max_step_size = min([max_step_size_x, max_step_size_z])

        if hasattr(P, 'comm'):
            max_step_size = P.comm.allreduce(max_step_size, op=MPI.MIN)
        return max_step_size

    def get_new_step_size(self, controller, step, **kwargs):
        if not CheckConvergence.check_convergence(step):
            return None

        L = step.levels[0]
        P = step.levels[0].prob

        L.sweep.compute_end_point()
        max_step_size = self.compute_max_step_size(P, L.uend)

        dt_new = L.status.dt_new if L.status.dt_new else max([self.params.dt_max, L.params.dt])
        L.status.dt_new = min([dt_new, self.params.cfl * max_step_size])
        L.status.dt_new = max([self.params.dt_min, L.status.dt_new])

        self.log(f'dt max: {max_step_size:.2e} -> New step size: {L.status.dt_new:.2e}', step)


class SpaceAdaptivity(ConvergenceController):
    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here.

        Default parameters are:
         - control_order (int): The order relative to other convergence controllers
         - dt_max (float): maximal step size
         - dt_min (float): minimal step size

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": 100,
            "nx_max": np.inf,
            "nx_min": 0,
            "nz_max": np.inf,
            "nz_min": 0,
            "factor": 3 / 2,
            "refinement_tol": 1e-8,
            "derefinement_tol": 1e-11,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def determine_restart(self, controller, S, *args, **kwargs):
        L = S.levels[0]
        P = L.prob

        if not CheckConvergence.check_convergence(S):
            return None

        L.sweep.compute_end_point()
        u_hat = P.transform(L.uend)

        if P.check_refinement_needed(
            u_hat, tol=self.params.refinement_tol, nx_max=self.params.nx_max, nz_max=self.params.nz_max
        ):
            # u_hat, P_new = P.refine_resolution(P.transform(L.u[0]), add_modes=self.params.refine_modes)
            # L.u[0] = P_new.u_init
            # L.u[0][:] = P_new.itransform(u_hat).real
            L.u[0], P_new = P.refine_resolution2(P.transform(L.u[0]), factor=self.params.factor)
            L.__dict__['_Level__prob'] = P_new
            L.status.dt_new = L.params.dt
            S.status.restart = True
            self.log(f"Restarting with refined resolution. New resolution: nx={L.prob.nx} nz={L.prob.nz}", S)
        elif (
            P.check_derefinement_ok(
                u_hat,
                tol=self.params.derefinement_tol,
                factor=self.params.factor,
                nx_min=self.params.nx_min,
                nz_min=self.params.nz_min,
            )
            and not S.status.restart
        ):
            for i in range(len(L.u)):
                # L.u[i], P_new = P.derefine_resolution(P.transform(L.u[i]), remove_modes=self.params.derefine_modes)
                u_hat, P_new = P.derefine_resolution2(L.u[i], factor=self.params.factor)
                L.u[i] = P_new.u_init
                L.u[i][:] = P_new.itransform(u_hat).real
            L.__dict__['_Level__prob'] = P_new
            self.log(f"Derefining resolution. New resolution: nx={L.prob.nx} nz={L.prob.nz}", S)
