import numpy as np
from mpi4py import MPI
from pmesh.pm import ParticleMesh

from pySDC.core.errors import ParameterError, ProblemError
from pySDC.core.problem import Problem
from pySDC.playgrounds.pmesh.PMESH_datatype_NEW import pmesh_datatype, rhs_imex_pmesh


class allencahn_imex(Problem):
    """
    Example implementing Allen-Cahn equation in 2-3D using PMESH for solving linear parts, IMEX time-stepping

    PMESH: https://github.com/rainwoodman/pmesh

    Attributes:
        xvalues: grid points in space
        dx: mesh width
    """

    def __init__(self, problem_params, dtype_u=pmesh_datatype, dtype_f=rhs_imex_pmesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: pmesh data type (will be passed to parent class)
            dtype_f: pmesh data type wuth implicit and explicit parts (will be passed to parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 1.0
        if 'init_type' not in problem_params:
            problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = None
        if 'dw' not in problem_params:
            problem_params['dw'] = 0.0

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'eps', 'L', 'radius', 'dw']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if not (isinstance(problem_params['nvars'], tuple) and len(problem_params['nvars']) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating ParticleMesh structure
        self.pm = ParticleMesh(
            BoxSize=problem_params['L'],
            Nmesh=list(problem_params['nvars']),
            dtype='f8',
            plan_method='measure',
            comm=problem_params['comm'],
        )

        # create test RealField to get the local dimensions (there's probably a better way to do that)
        tmp = self.pm.create(type='real')
        tmps = tmp.r2c()

        # invoke super init, passing the communicator and the local dimensions as init
        super(allencahn_imex, self).__init__(
            init=(self.pm.comm, tmps.value.shape), dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params
        )

        # Need this for diagnostics
        self.dx = self.params.L / problem_params['nvars'][0]
        self.dy = self.params.L / problem_params['nvars'][1]
        self.xvalues = [i * self.dx - problem_params['L'] / 2 for i in range(problem_params['nvars'][0])]
        self.yvalues = [i * self.dy - problem_params['L'] / 2 for i in range(problem_params['nvars'][1])]

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        def Laplacian(k, v):
            k2 = sum(ki**2 for ki in k)
            return -k2 * v

        f = self.dtype_f(self.init)
        tmp_u = self.pm.create(type='complex', value=u.values)
        f.impl.values = tmp_u.apply(Laplacian).value

        if self.params.eps > 0:
            tmp_u = tmp_u.c2r(out=Ellipsis)
            tmp_f = -2.0 / self.params.eps**2 * tmp_u * (1.0 - tmp_u) * (
                1.0 - 2.0 * tmp_u
            ) - 6.0 * self.params.dw * tmp_u * (1.0 - tmp_u)
            f.expl.values = tmp_f.r2c(out=Ellipsis).value

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        def linear_solve(k, v):
            k2 = sum(ki**2 for ki in k)
            return 1.0 / (1.0 + factor * k2) * v

        me = self.dtype_u(self.init)
        tmp_rhs = self.pm.create(type='complex', value=rhs.values)
        me.values = tmp_rhs.apply(linear_solve, out=Ellipsis).value
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        def circle(i, v):
            r = [ii * (Li / ni) - 0.5 * Li for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
            r2 = sum(ri**2 for ri in r)
            return 0.5 * (1.0 + np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps)))

        def circle_rand(i, v):
            L = [int(l) for l in v.BoxSize]
            r = [ii * (Li / ni) - 0.5 * Li for ii, ni, Li in zip(i, v.Nmesh, L)]
            rshift = r.copy()
            ndim = len(r)
            data = 0
            # get random radii for circles/spheres
            np.random.seed(1)
            lbound = 3.0 * self.params.eps
            ubound = 0.5 - self.params.eps
            rand_radii = (ubound - lbound) * np.random.random_sample(size=tuple(L)) + lbound
            # distribnute circles/spheres
            if ndim == 2:
                for indexi, i in enumerate(range(-L[0] + 1, L[0], 2)):
                    for indexj, j in enumerate(range(-L[1] + 1, L[1], 2)):
                        # shift x and y coordinate depending on which box we are in
                        rshift[0] = r[0] + i / 2
                        rshift[1] = r[1] + j / 2
                        # build radius
                        r2 = sum(ri**2 for ri in rshift)
                        # add this blob, shifted by 1 to avoid issues with adding up negative contributions
                        data += np.tanh((rand_radii[indexi, indexj] - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps)) + 1
            # get rid of the 1
            data *= 0.5
            assert np.all(data <= 1.0)
            return data

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init)
        if self.params.init_type == 'circle':
            tmp_u = self.pm.create(type='real', value=0.0)
            tmp_u.apply(circle, kind='index', out=Ellipsis)
            me.values = tmp_u.r2c().value
        elif self.params.init_type == 'circle_rand':
            tmp_u = self.pm.create(type='real', value=0.0)
            tmp_u.apply(circle_rand, kind='index', out=Ellipsis)
            me.values = tmp_u.r2c().value
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)

        return me


class allencahn_imex_timeforcing(allencahn_imex):
    """
    Example implementing Allen-Cahn equation in 2-3D using PMESH for solving linear parts, IMEX time-stepping,
    time-dependent forcing
    """

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        def Laplacian(k, v):
            k2 = sum(ki**2 for ki in k)
            return -k2 * v

        f = self.dtype_f(self.init)
        tmp_u = self.pm.create(type='real', value=u.values)
        f.impl.values = tmp_u.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis).value

        if self.params.eps > 0:
            f.expl.values = -2.0 / self.params.eps**2 * u.values * (1.0 - u.values) * (1.0 - 2.0 * u.values)

        # build sum over RHS without driving force
        Rt_local = f.impl.values.sum() + f.expl.values.sum()
        if self.pm.comm is not None:
            Rt_global = self.pm.comm.allreduce(sendobj=Rt_local, op=MPI.SUM)
        else:
            Rt_global = Rt_local

        # build sum over driving force term
        Ht_local = np.sum(6.0 * u.values * (1.0 - u.values))
        if self.pm.comm is not None:
            Ht_global = self.pm.comm.allreduce(sendobj=Ht_local, op=MPI.SUM)
        else:
            Ht_global = Rt_local

        # add/substract time-dependent driving force
        dw = Rt_global / Ht_global
        f.expl.values -= 6.0 * dw * u.values * (1.0 - u.values)

        return f


class allencahn_imex_stab(allencahn_imex):
    """
    Example implementing Allen-Cahn equation in 2-3D using PMESH for solving linear parts, IMEX time-stepping with
    stabilized splitting
    """

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        def Laplacian(k, v):
            k2 = sum(ki**2 for ki in k) + 1.0 / self.params.eps**2
            return -k2 * v

        f = self.dtype_f(self.init)
        tmp_u = self.pm.create(type='real', value=u.values)
        f.impl.values = tmp_u.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis).value
        if self.params.eps > 0:
            f.expl.values = (
                -2.0 / self.params.eps**2 * u.values * (1.0 - u.values) * (1.0 - 2.0 * u.values)
                - 6.0 * self.params.dw * u.values * (1.0 - u.values)
                + 1.0 / self.params.eps**2 * u.values
            )
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        def linear_solve(k, v):
            k2 = sum(ki**2 for ki in k) + 1.0 / self.params.eps**2
            return 1.0 / (1.0 + factor * k2) * v

        me = self.dtype_u(self.init)
        tmp_rhs = self.pm.create(type='real', value=rhs.values)
        me.values = tmp_rhs.r2c().apply(linear_solve, out=Ellipsis).c2r(out=Ellipsis).value

        return me
