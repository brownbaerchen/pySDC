from pySDC.core.problem import Problem
from pySDC.helpers.problem_helper import get_finite_difference_matrix, get_1d_grid
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
from qmat.lagrange import getSparseInterpolationMatrix
import numpy as np
import matplotlib.pyplot as plt


class Burgers1DSL(Problem):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, N=64, L=2 * np.pi, nu=1e-2):

        super().__init__(init=((N,), None, np.dtype('float64')))
        self._makeAttributeAndRegister('N', 'L', 'nu', localVars=locals())

        h, self.grid = get_1d_grid(N, 'periodic', 0, L)
        self.Id = sp.eye(N)
        self.dx, _ = get_finite_difference_matrix(1, 8, stencil_type='center', bc='periodic', dx=h, dim=1, size=N)
        self.dxx, _ = get_finite_difference_matrix(2, 8, stencil_type='center', bc='periodic', dx=h, dim=1, size=N)

    def get_departure_points(self, u, dt):
        return self.grid - u * dt

    def interpolate(self, u, departure_points, order=8):
        me = self.u_init
        I = getSparseInterpolationMatrix(self.grid, departure_points, order=order, gridPeriod=self.L)
        me[...] = I @ u
        return me

    def eval_f(self, u, *args):
        f = self.f_init

        f[...] = self.nu * self.dxx @ u

        return f

    def solve_system(self, rhs, dt, u0=None, t=None):
        me = self.u_init

        me[...] = spsolve(self.Id - dt * self.nu * self.dxx, rhs)
        return me

    def u_exact(self, t=0):
        me = self.u_init
        me[...] = np.sin(self.grid * 2 * np.pi / self.L)
        return me

    def get_fig(self):
        fig, ax = plt.subplots()
        return fig

    def plot(self, u, t=-1, fig=None, **kwargs):
        fig = self.get_fig() if fig is None else fig
        ax = fig.get_axes()[0]
        ax.plot(self.grid, u, label=f't={t:.2f}', **kwargs)
        ax.legend(frameon=False)


class Burgers1DIMEX(Burgers1DSL):
    dtype_f = imex_mesh

    def eval_f(self, u, *args):
        f = self.f_init

        f.impl[...] = self.nu * self.dxx @ u
        f.expl[...] = -u * (self.dx @ u)

        return f


class AdvectionDiffusion1DSL(Burgers1DSL):
    c = 1

    def get_departure_points(self, u, dt):
        return self.grid - self.c * dt

    def u_exact(self, t=0):
        me = self.u_init
        me[...] = np.sin((self.grid - self.c * t) * 2 * np.pi / self.L) * np.exp(-self.nu * t)
        return me


class AdvectionDiffusion1D(AdvectionDiffusion1DSL):
    def eval_f(self, u, *args):
        f = self.f_init

        f[...] = (-self.c * self.dx + self.nu * self.dxx) @ u

        return f

    def solve_system(self, rhs, dt, u0=None, t=None):
        me = self.u_init

        me[...] = spsolve(self.Id - dt * (-self.c * self.dx + self.nu * self.dxx), rhs)
        return me


class AdvectionDiffusion1DIMEX(AdvectionDiffusion1DSL):
    dtype_f = imex_mesh

    def eval_f(self, u, *args):
        f = self.f_init

        f.impl[...] = self.nu * self.dxx @ u
        f.expl[...] = -self.c * self.dx @ u

        return f
