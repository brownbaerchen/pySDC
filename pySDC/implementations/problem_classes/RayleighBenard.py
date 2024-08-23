import numpy as np

from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class RayleighBenard(GenericSpectralLinear):

    dtype_u = mesh
    dtype_f = imex_mesh

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
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

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

        kappa = (Rayleigh * Prandl) ** (-1 / 2.0)
        nu = (Rayleigh / Prandl) ** (-1 / 2.0)

        # construct operators
        L_lhs = {
            'vz': {'v': -Dz, 'vz': I},  # algebraic constraint for first derivative
            'uz': {'u': -Dz, 'uz': I},  # algebraic constraint for first derivative
            'Tz': {'T': -Dz, 'Tz': I},  # algebraic constraint for first derivative
            'p': {'u': Dx, 'vz': I},  # divergence free constraint
            'u': {'p': Dx, 'u': -nu * Dxx, 'uz': -nu * Dz},
            'v': {'p': Dz, 'v': -nu * Dxx, 'vz': -nu * Dz, 'T': -I},
            'T': {'T': -kappa * Dxx, 'Tz': -kappa * Dz},
        }
        self.setup_L(L_lhs)

        # mass matrix
        M_lhs = {i: {i: I} for i in ['u', 'v', 'T']}
        self.setup_M(M_lhs)

        self.add_BC(
            component='p',
            equation='p',
            axis=0,
            v=self.BCs['p_integral'],
            kind='integral',
            line=0,
            pressure_gauge_FFT=True,
        )
        # self.add_BC(
        #     component='p', equation='p', axis=1, v=self.BCs['p_integral'], kind='integral', line=-1
        # )
        self.add_BC(component='T', equation='T', axis=1, x=-1, v=self.BCs['T_bottom'], kind='Dirichlet')
        self.add_BC(component='T', equation='Tz', axis=1, x=1, v=self.BCs['T_top'], kind='Dirichlet', line=-1)
        self.add_BC(component='v', equation='v', axis=1, x=-1, v=self.BCs['v_bottom'], kind='Dirichlet')
        self.add_BC(component='v', equation='vz', axis=1, x=1, v=self.BCs['v_top'], kind='Dirichlet', line=-1)
        self.add_BC(component='u', equation='u', axis=1, v=self.BCs['u_top'], x=1, kind='Dirichlet')
        self.add_BC(
            component='u',
            equation='uz',
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

        if compute_violations:
            f_impl_hat[iuz][:] = (Dz @ u_hat[iu].flatten()).reshape(shape) - u_hat[iuz]
            f_impl_hat[ivz][:] = (Dz @ u_hat[iv].flatten()).reshape(shape) - u_hat[ivz]
            f_impl_hat[iTz][:] = (Dz @ u_hat[iT].flatten()).reshape(shape) - u_hat[iTz]
            f_impl_hat[ip][:] = (Dz @ u_hat[iv].flatten() + Dx @ u_hat[iu].flatten()).reshape(shape)

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
        noise[iT] = rng.normal(size=me[self.iT].shape)

        Kz, Kx = self.get_wavenumbers()
        kzmax = self.nz - 3 if kzmax is None else kzmax
        kxmax = self.nx // 2 if kxmax is None else kxmax
        noise_hat = self.u_init_forward
        noise_hat[:] = rng.normal(size=noise_hat[self.iT].shape)
        noise_hat[iT, np.abs(Kx) > kxmax] *= 0
        noise_hat[iT, Kz > kzmax] *= 0
        noise = self.itransform(noise_hat)

        xp = self.xp
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

    def compute_vorticity(self, u):
        u_hat = self.transform(u)
        Dz = self.U2T @ self.Dz
        Dx = self.U2T @ self.Dx
        iu, iv = self.index(['u', 'v'])

        vorticity_hat = self.u_init_forward
        vorticity_hat[0] = (Dx * u_hat[iv].flatten() + Dz @ u_hat[iu].flatten()).reshape(u[iu].shape)
        return self.itransform(vorticity_hat)[0].real

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

        vmin = u.min()
        vmax = u.max()

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
