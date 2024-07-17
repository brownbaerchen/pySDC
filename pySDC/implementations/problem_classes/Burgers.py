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

        S = len(self.helper.components)  # number of variables

        super().__init__(init=((S, N), None, np.dtype('complex128')))

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

    def u_exact(self, *args, **kwargs):
        me = self.u_init
        me[self.helper.index('u')][:] = ((self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x) * np.cos(
            self.x * np.pi * self.f
        )
        me[self.helper.index('ux')][:] = (self.BCr - self.BCl) / 2 * np.cos(self.x * np.pi * self.f) + (
            (self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x
        ) * self.f * np.pi * -np.sin(self.x * np.pi * self.f)
        return me

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iux = self.helper.index('u'), self.helper.index('ux')

        u_hat = self.helper.transform(u, axes=(-1,))
        f.impl[iu] = -self.epsilon * self.helper.itransform(
            (self.C @ self.Dx @ u_hat[iux].flatten()).reshape(u_hat[iu].shape), axes=(-1,)
        )
        f.expl[iu] = u[iu] * u[iux]
        return f

    def solve_system(self, rhs, factor, *args, **kwargs):
        sol = self.u_init

        _rhs = (self.M @ rhs.flatten()).reshape(sol.shape)
        _rhs = self.helper.put_BCs_in_rhs(_rhs)
        rhs_hat = self.helper.transform(_rhs, axes=(-1,))

        A = self.M + factor * self.L
        A = self.helper.put_BCs_in_matrix(A)

        sol_hat = (self.helper.sparse_lib.linalg.spsolve(A.tocsc(), rhs_hat.flatten())).reshape(sol.shape)

        for i in range(sol.shape[0]):
            sol_hat[i] = self.C @ sol_hat[i]

        sol[:] = self.helper.itransform(sol_hat, axes=(-1,))
        return sol
