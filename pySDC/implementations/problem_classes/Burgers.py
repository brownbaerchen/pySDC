import numpy as np

from pySDC.core.problem import Problem
from pySDC.helpers.spectral_helper import SpectralHelper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class Burgers1D(Problem):

    dtype_u = mesh
    dtype_f = imex_mesh
    xp = np

    def __init__(self, N=64, epsilon=0.1, BCl=1, BCr=-1, f=0, mode='T2U'):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        self.helper = SpectralHelper()
        self.helper.add_axis(base='cheby', N=N, mode=mode)
        self.helper.add_component(['u', 'ux'])
        self.helper.setup_fft()

        super().__init__(init=self.helper.init)

        self.x = self.helper.get_grid()[0]

        # prepare matrices
        Dx = self.helper.get_differentiation_matrix(axes=(0,))
        I = self.helper.get_Id()
        self.Dx = Dx
        self.C = self.helper.get_basis_change_matrix()

        # construct linear operator
        L = self.helper.get_empty_operator_matrix()
        self.helper.add_equation_lhs(L, 'u', {'ux': -epsilon * Dx})
        self.helper.add_equation_lhs(L, 'ux', {'u': -Dx, 'ux': I})
        self.L = self.helper.convert_operator_matrix_to_operator(L)

        # construct mass matrix
        M = self.helper.get_empty_operator_matrix()
        self.helper.add_equation_lhs(M, 'u', {'u': I})
        self.M = self.helper.convert_operator_matrix_to_operator(M)

        # boundary conditions
        self.helper.add_BC(component='u', equation='u', axis=0, x=1, v=BCr)
        self.helper.add_BC(component='u', equation='ux', axis=0, x=-1, v=BCl)
        self.helper.setup_BCs()

    def u_exact(self, t=0, *args, **kwargs):
        me = self.u_init

        # x = (self.x + 1) / 2
        # g = 4 * (1 + np.exp(-(4 * x + t)/self.epsilon/32))
        # g_x = 4 * np.exp(-(4 * x + t)/self.epsilon/32) * (-4/self.epsilon/32)

        # me[0] = 3./4. - 1./g
        # me[1] = 1/g**2 * g_x

        # return me

        if t == 0:
            me[self.helper.index('u')][:] = ((self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x) * np.cos(
                self.x * np.pi * self.f
            )
            me[self.helper.index('ux')][:] = (self.BCr - self.BCl) / 2 * np.cos(self.x * np.pi * self.f) + (
                (self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x
            ) * self.f * np.pi * -np.sin(self.x * np.pi * self.f)
        elif t == np.inf and self.f == 0 and self.BCl == -self.BCr:
            me[0] = (self.BCl * np.exp((self.BCl - self.BCr) / (2 * self.epsilon) * self.x) + self.BCr) / (
                np.exp((self.BCl - self.BCr) / (2 * self.epsilon) * self.x) + 1
            )
        else:
            raise NotImplementedError

        return me

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iux = self.helper.index('u'), self.helper.index('ux')

        u_hat = self.helper.transform(u, axes=(-1,))

        Dx_u_hat = self.u_init
        Dx_u_hat[iu] = (self.C @ self.Dx @ u_hat[iux].flatten()).reshape(u_hat[iu].shape)

        f.impl[iu] = -self.epsilon * self.helper.itransform(Dx_u_hat, axes=(-1,))[iu]
        f.expl[iu] = u[iu] * u[iux]
        return f

    def solve_system(self, rhs, factor, *args, **kwargs):
        sol = self.u_init

        rhs_hat = self.helper.transform(rhs, axes=(-1,))
        rhs_hat = (self.M @ rhs_hat.flatten()).reshape(sol.shape)
        rhs_hat = self.helper.put_BCs_in_rhs(rhs_hat, istransformed=True)

        A = self.M + factor * self.L
        A = self.helper.put_BCs_in_matrix(A)

        sol_hat = (self.helper.sparse_lib.linalg.spsolve(A, rhs_hat.flatten())).reshape(sol.shape)

        sol[:] = self.helper.itransform(sol_hat, axes=(-1,))
        return sol


class Burgers2D(Problem):
    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nx=64, nz=64, epsilon=0.1, mode='T2U'):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        self.helper = SpectralHelper()
        self.helper.add_axis(base='fft', N=nx)
        self.helper.add_axis(base='cheby', N=nz, mode=mode)
        self.helper.add_component(['u', 'v', 'ux', 'vz'])
        self.helper.setup_fft()

        super().__init__(init=self.helper.init)

        self.Z, self.X = self.helper.get_grid()

        # prepare matrices
        Dx = -self.helper.get_differentiation_matrix(axes=(0,))
        Dz = self.helper.get_differentiation_matrix(axes=(1,))
        I = self.helper.get_Id()
        self.Dx = Dx
        self.Dz = Dz
        self.C = self.helper.get_basis_change_matrix()

        # construct linear operator
        L = self.helper.get_empty_operator_matrix()
        self.helper.add_equation_lhs(L, 'u', {'ux': -epsilon * Dx})
        self.helper.add_equation_lhs(L, 'v', {'vz': -epsilon * Dz})
        self.helper.add_equation_lhs(L, 'ux', {'u': -Dx, 'ux': I})
        self.helper.add_equation_lhs(L, 'vz', {'v': -Dz, 'vz': I})
        self.L = self.helper.convert_operator_matrix_to_operator(L)

        # construct mass matrix
        M = self.helper.get_empty_operator_matrix()
        self.helper.add_equation_lhs(M, 'u', {'u': I})
        self.M = self.helper.convert_operator_matrix_to_operator(M)

        # boundary conditions
        # self.helper.add_BC(component='u', equation='ux', axis=0, x=-1, v=1)
        self.helper.add_BC(component='v', equation='v', axis=1, x=1, v=-1)
        self.helper.add_BC(component='v', equation='vz', axis=1, x=-1, v=1)
        self.helper.setup_BCs()

    def u_exact(self, t=0, *args, **kwargs):
        me = self.u_init

        # g = 4 * (1 + np.exp(-(4 * self.x + t)/self.epsilon/32))
        # g_x = 4 * np.exp(-(4 * self.x + t)/self.epsilon/32) * (-4/self.epsilon/32)

        # me[0] = 3./4. - 1./g
        # me[1] = 1/g**2 * g_x

        # return me

        if t == 0:
            me[self.helper.index('u')][:] = ((self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x) * np.cos(
                self.x * np.pi * self.f
            )
            me[self.helper.index('ux')][:] = (self.BCr - self.BCl) / 2 * np.cos(self.x * np.pi * self.f) + (
                (self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x
            ) * self.f * np.pi * -np.sin(self.x * np.pi * self.f)
        elif t == np.inf and self.f == 0 and self.BCl == -self.BCr:
            me[0] = (self.BCl * np.exp((self.BCl - self.BCr) / (2 * self.epsilon) * self.x) + self.BCr) / (
                np.exp((self.BCl - self.BCr) / (2 * self.epsilon) * self.x) + 1
            )
        else:
            raise NotImplementedError

        return me

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iv, iux, ivz = (self.helper.index(comp) for comp in self.helper.components)

        u_hat = self.helper.transform(u, axes=(-1, -2))
        f_hat = self.u_init
        f_hat[iu] = -self.epsilon * (self.C @ self.Dx @ u_hat[iux].flatten()).reshape(u_hat[iux].shape)
        f_hat[iv] = -self.epsilon * (self.C @ self.Dz @ u_hat[ivz].flatten()).reshape(u_hat[iux].shape)
        f.impl[...] = self.helper.itransform(f_hat, axes=(-2, -1))

        f.expl[iu] = u[iu] * u[iux]
        f.expl[iv] = u[iv] * u[ivz]
        return f

    def solve_system(self, rhs, factor, *args, **kwargs):
        sol = self.u_init

        rhs_hat = self.helper.transform(rhs, axes=(-1,))
        rhs_hat = (self.M @ rhs_hat.flatten()).reshape(sol.shape)
        rhs_hat = self.helper.put_BCs_in_rhs(rhs_hat, istransformed=True)

        A = self.M + factor * self.L
        A = self.helper.put_BCs_in_matrix(A)

        sol_hat = (self.helper.sparse_lib.linalg.spsolve(A, rhs_hat.flatten())).reshape(sol.shape)

        sol[:] = self.helper.itransform(sol_hat, axes=(-1,))
        return sol
