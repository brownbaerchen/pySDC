import numpy as np
from scipy import sparse as sp

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear

from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper


class Heat1DChebychov(GenericSpectralLinear):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, a=0, b=0, f=1, nu=1.0, **kwargs):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        bases = [{'base': 'chebychov', 'N': nvars}]
        components = ['u', 'ux']

        super().__init__(bases, components, **kwargs)

        self.x = self.get_grid()[0]

        I = self.get_Id()
        Dx = self.get_differentiation_matrix(axes=(0,))
        self.Dx = Dx
        self.U2T = self.get_basis_change_matrix()

        L_lhs = {
            'ux': {'u': -Dx, 'ux': I},
            'u': {'ux': -nu * Dx},
        }
        self.setup_L(L_lhs)

        M_lhs = {'u': {'u': I}}
        self.setup_M(M_lhs)

        self.add_BC(component='u', equation='u', axis=0, x=-1, v=a, kind="Dirichlet")
        self.add_BC(component='u', equation='ux', axis=0, x=1, v=b, kind="Dirichlet")
        self.setup_BCs()

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iux = self.index(self.components)

        me_hat = self.u_init_forward
        me_hat[:] = self.transform(u)
        me_hat[iu] = (self.nu * self.U2T @ self.Dx @ me_hat[iux].flatten()).reshape(me_hat[iu].shape)
        me = self.itransform(me_hat).real

        f[self.index("u")] = me[iu]
        return f

    def u_exact(self, t):
        xp = self.xp
        iu, iux = self.index(self.components)
        u = self.u_init

        u[iu] = (
            xp.sin(self.f * np.pi * self.x) * xp.exp(-self.nu * (self.f * np.pi) ** 2 * t)
            + (self.b - self.a) / 2 * self.x
            + (self.b + self.a) / 2
        )
        u[iux] = (
            self.f * np.pi * xp.cos(self.f * np.pi * self.x) * xp.exp(-self.nu * (self.f * np.pi) ** 2 * t)
            + (self.b - self.a) / 2
        )

        return u


class Heat2DChebychov(GenericSpectralLinear):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nx=128, ny=128, base_x='fft', base_y='chebychov', a=0, b=0, c=0, fx=1, fy=1, nu=1.0, **kwargs):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        bases = [{'base': base_x, 'N': nx}, {'base': base_y, 'N': ny}]
        components = ['u', 'ux', 'uy']

        super().__init__(bases, components, **kwargs)

        self.Y, self.X = self.get_grid()

        I = self.get_Id()
        self.Dx = self.get_differentiation_matrix(axes=(0,))
        self.Dy = self.get_differentiation_matrix(axes=(1,))
        self.U2T = self.get_basis_change_matrix()

        L_lhs = {
            'ux': {'u': -self.Dx, 'ux': I},
            'uy': {'u': -self.Dy, 'uy': I},
            'u': {'ux': -nu * self.Dx, 'uy': -nu * self.Dy},
        }
        self.setup_L(L_lhs)

        M_lhs = {'u': {'u': I}}
        self.setup_M(M_lhs)

        for base in [base_x, base_y]:
            assert base in ['chebychov', 'fft']

        alpha = (self.b - self.a) / 2.0
        beta = (self.c - self.b) / 2.0
        gamma = (self.c + self.a) / 2.0

        if base_x == 'chebychov':
            y = self.Y[0, :]
            if self.base_y == 'fft':
                self.add_BC(component='u', equation='u', axis=0, x=-1, v=beta * y - alpha + gamma, kind='Dirichlet')
            self.add_BC(
                component='u', equation='ux', axis=0, x=1, v=beta * y + alpha + gamma, kind='Dirichlet', zero_line=False
            )
        else:
            assert a == b, f'Need periodic boundary conditions in x for {base_x} method!'
        if base_y == 'chebychov':
            x = self.X[:, 0]
            self.add_BC(
                component='u', equation='u', axis=1, x=-1, v=alpha * x - beta + gamma, kind='Dirichlet', zero_line=True
            )
            self.add_BC(
                component='u', equation='uy', axis=1, x=1, v=alpha * x + beta + gamma, kind='Dirichlet', zero_line=True
            )
        else:
            assert c == b, f'Need periodic boundary conditions in y for {base_y} method!'
        self.setup_BCs()

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iux, iuy = self.index(self.components)

        me_hat = self.u_init_forward
        me_hat[:] = self.transform(u)
        me_hat[iu] = self.nu * (
            self.U2T @ self.Dx @ me_hat[iux].flatten() + self.U2T @ self.Dy @ me_hat[iuy].flatten()
        ).reshape(me_hat[iu].shape)
        me = self.itransform(me_hat)

        f[self.index("u")] = me[iu].real
        return f

    def u_exact(self, t):
        xp = self.xp
        iu, iux, iuy = self.index(self.components)
        u = self.u_init

        fx = self.fx if self.base_x == 'fft' else np.pi * self.fx
        fy = self.fy if self.base_y == 'fft' else np.pi * self.fy

        time_dep = xp.exp(-self.nu * (fx**2 + fy**2) * t)

        alpha = (self.b - self.a) / 2.0
        beta = (self.c - self.b) / 2.0
        gamma = (self.c + self.a) / 2.0

        u[iu] = xp.sin(fx * self.X) * xp.sin(fy * self.Y) * time_dep + alpha * self.X + beta * self.Y + gamma
        u[iux] = fx * xp.cos(fx * self.X) * xp.sin(fy * self.Y) * time_dep + alpha
        u[iuy] = fy * xp.sin(fx * self.X) * xp.cos(fy * self.Y) * time_dep + beta

        return u