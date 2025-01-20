from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh, IMEX_firedrake_mesh
import firedrake as fd
import numpy as np
from mpi4py import MPI


class Heat1DForcedFiredrake(Problem):
    r"""
    Example implementing the forced one-dimensional heat equation with Dirichlet boundary conditions

    .. math::
        \frac{d u}{d t} = \nu \frac{d^2 u}{d x^2} + f

    for :math:`x \in \Omega:=[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = -\sin(\pi x) (\sin(t) - \nu \pi^2 \cos(t)).

    For initial conditions with constant c

    .. math::
        u(x, 0) = \sin(\pi x) + c,

    the exact solution is given by

    .. math::
        u(x, t) = \sin(\pi x)\cos(t) + c.

    Here, the problem is discretized with finite elements using firedrake. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    We invert the Laplacian implicitly and treat the forcing term explicitly.
    The solvers for the arising variational problems are cached for multiple collocation nodes and step sizes.

    Parameters
    ----------
    nvars : int, optional
        Spatial resolution, i.e., numbers of degrees of freedom in space.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    c: float, optional
        Constant for the Dirichlet boundary condition :math: `c`
    LHS_cache_size : int, optional
        Cache size for variational problem solvers
    comm : MPI communicator, optional
        Supply an MPI communicator for spatial parallelism
    """

    dtype_u = firedrake_mesh
    dtype_f = IMEX_firedrake_mesh

    def __init__(self, n=30, nu=0.1, c=0.0, LHS_cache_size=12, comm=None):
        comm = MPI.COMM_WORLD if comm is None else comm
        self.mesh = fd.UnitIntervalMesh(n, comm=comm)
        self.V = fd.FunctionSpace(self.mesh, "CG", 4)

        super().__init__(self.V)
        self._makeAttributeAndRegister('n', 'nu', 'c', 'LHS_cache_size', 'comm', localVars=locals(), readOnly=True)

        # prepare caches and IO variables for solvers
        self.solvers = {}
        self.tmp_in = fd.Function(self.V)
        self.tmp_out = fd.Function(self.V)

        self.work_counters['solver_setup'] = WorkCounter()
        self.work_counters['solves'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        """
        Evaluate the right hand side.
        The forcing term is simply interpolated to the grid.
        The Laplacian is evaluated via a variational problem, where the mass matrix is inverted and homogeneous boundary conditions are applied.

        Parameters
        ----------
        u : dtype_u
            Solution at which to evaluate
        t : float
            Time at which to evaluate

        Returns
        -------
        f : dtype_f
            The evaluated right hand side
        """
        if not hasattr(self, '__solv_eval_f_implicit'):
            v = fd.TestFunction(self.V)
            u_trial = fd.TrialFunction(self.V)

            a = u_trial * v * fd.dx
            L_impl = -fd.inner(self.nu * fd.nabla_grad(self.tmp_in), fd.nabla_grad(v)) * fd.dx

            bcs = [fd.bcs.DirichletBC(self.V, fd.Constant(0), area) for area in [1, 2]]

            prob = fd.LinearVariationalProblem(a, L_impl, self.tmp_out, bcs=bcs)
            self.__solv_eval_f_implicit = fd.LinearVariationalSolver(prob)

        self.tmp_in.assign(u.functionspace)

        self.__solv_eval_f_implicit.solve()

        me = self.dtype_f(self.init)
        me.impl.assign(self.tmp_out)

        x = fd.SpatialCoordinate(self.mesh)
        me.expl.interpolate(-(np.sin(t) - self.nu * np.pi**2 * np.cos(t)) * fd.sin(np.pi * x[0]))

        self.work_counters['rhs']()

        return me

    def solve_system(self, rhs, factor, *args, **kwargs):
        r"""
        Linear solver for :math:`(M - factor nu * Lap) u = rhs`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time.

        Returns
        -------
        u : dtype_u
            Solution.
        """

        if factor not in self.solvers.keys():
            if len(self.solvers) >= self.LHS_cache_size:
                self.solvers.pop(list(self.solvers.keys())[0])

            u = fd.TrialFunction(self.V)
            v = fd.TestFunction(self.V)

            a = u * v * fd.dx + fd.Constant(factor) * fd.inner(self.nu * fd.nabla_grad(u), fd.nabla_grad(v)) * fd.dx
            L = fd.inner(self.tmp_in, v) * fd.dx

            bcs = [fd.bcs.DirichletBC(self.V, fd.Constant(self.c), area) for area in [1, 2]]

            prob = fd.LinearVariationalProblem(a, L, self.tmp_out, bcs=bcs)
            self.solvers[factor] = fd.LinearVariationalSolver(prob)

            self.work_counters['solver_setup']()

        self.tmp_in.assign(rhs.functionspace)
        self.tmp_out.assign(rhs.functionspace)
        self.solvers[factor].solve()
        me = self.dtype_u(self.init)
        me.assign(self.tmp_out)
        self.work_counters['solves']()
        return me

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """
        me = self.u_init
        x = fd.SpatialCoordinate(self.mesh)
        me.interpolate(np.cos(t) * fd.sin(np.pi * x[0]) + self.c)
        return me
