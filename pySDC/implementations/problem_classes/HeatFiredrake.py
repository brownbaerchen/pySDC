from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh, IMEX_firedrake_mesh
import firedrake as fd
import numpy as np


class firedrake_heat(Problem):
    dtype_u = firedrake_mesh
    dtype_f = IMEX_firedrake_mesh

    def __init__(self, n=30, nu=0.1, c=0.0, LHS_cache_size=12):
        self.mesh = fd.UnitIntervalMesh(n)
        self.V = fd.FunctionSpace(self.mesh, "CG", 4)

        super().__init__(self.V)
        self._makeAttributeAndRegister('n', 'nu', 'c', 'LHS_cache_size', localVars=locals(), readOnly=True)

        self.solvers = {}
        self.rhs = {}
        self.tmp = fd.Function(self.V)

        self.work_counters['solver_setup'] = WorkCounter()
        self.work_counters['solves'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        if not hasattr(self, '__solv_eval_f_implicit'):
            _u = u.functionspace
            v = fd.TestFunction(self.V)
            u_trial = fd.TrialFunction(self.V)

            a = u_trial * v * fd.dx
            L_impl = -fd.inner(self.nu * fd.nabla_grad(_u), fd.nabla_grad(v)) * fd.dx

            bcs = [fd.bcs.DirichletBC(self.V, fd.Constant(0), area) for area in [1, 2]]

            prob = fd.LinearVariationalProblem(a, L_impl, self.tmp, bcs=bcs)
            self.__solv_eval_f_implicit = fd.LinearVariationalSolver(prob)

        self.__solv_eval_f_implicit.solve()

        me = self.dtype_f(self.init)
        me.impl.assign(self.tmp)

        x = fd.SpatialCoordinate(self.mesh)
        me.expl.interpolate(-(np.sin(t) - self.nu * np.pi**2 * np.cos(t)) * fd.sin(np.pi * x[0]))

        self.work_counters['rhs']()

        return me

    def solve_system(self, rhs, factor, *args, **kwargs):

        if factor not in self.solvers.keys():
            if len(self.solvers) >= self.LHS_cache_size:
                self.rhs.pop(list(self.solvers.keys())[0])
                self.solvers.pop(list(self.solvers.keys())[0])

            self.rhs[factor] = fd.Function(self.V)

            u = fd.TrialFunction(self.V)
            v = fd.TestFunction(self.V)

            a = u * v * fd.dx + fd.Constant(factor) * fd.inner(self.nu * fd.nabla_grad(u), fd.nabla_grad(v)) * fd.dx
            L = fd.inner(self.rhs[factor], v) * fd.dx

            bcs = [fd.bcs.DirichletBC(self.V, fd.Constant(self.c), area) for area in [1, 2]]

            prob = fd.LinearVariationalProblem(a, L, self.tmp, bcs=bcs)
            self.solvers[factor] = fd.LinearVariationalSolver(prob)

            self.work_counters['solver_setup']()

        self.rhs[factor].assign(rhs.functionspace)
        self.solvers[factor].solve()
        me = self.dtype_u(self.init)
        me.assign(self.tmp)
        self.work_counters['solves']()
        return me

    def u_exact(self, t):
        me = self.u_init
        x = fd.SpatialCoordinate(self.mesh)
        me.interpolate(np.cos(t) * fd.sin(np.pi * x[0]) + self.c)
        return me
