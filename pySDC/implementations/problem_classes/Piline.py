import numpy as np
from scipy.integrate import solve_ivp

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class piline(ptype):
    """
    Example implementing the Piline model as in the description in the PinTSimE project

    Attributes:
        A: system matrix, representing the 3 ODEs
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, Vs, Rs, C1, Rpi, Lpi, C2, Rl):
        """Initialization routine"""

        nvars = 3
        # invoke super init, passing number of dofs
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'Vs', 'Rs', 'C1', 'Rpi', 'Lpi', 'C2', 'Rl', localVars=locals(), readOnly=True
        )

        # compute dx and get discretization matrix A
        self.A = np.zeros((3, 3))
        self.A[0, 0] = -1 / (self.Rs * self.C1)
        self.A[0, 2] = -1 / self.C1
        self.A[1, 1] = -1 / (self.Rl * self.C2)
        self.A[1, 2] = 1 / self.C2
        self.A[2, 0] = 1 / self.Lpi
        self.A[2, 1] = -1 / self.Lpi
        self.A[2, 2] = -self.Rpi / self.Lpi

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)
        f.expl[0] = self.Vs / (self.Rs * self.C1)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to approximate the exact solution at time t by scipy

        Args:
            t (float): current time
            u_init (pySDC.problem.Piline.dtype_u): initial conditions for getting the exact solution
            t_init (float): the starting time

        Returns:
            dtype_u: exact solution (kind of)
        """

        me = self.dtype_u(self.init)

        # fill initial conditions
        me[0] = 0.0  # v1
        me[1] = 0.0  # v2
        me[2] = 0.0  # p3

        if t > 0.0:
            if u_init is not None:
                if t_init is None:
                    raise ValueError(
                        'Please supply `t_init` when you want to get the exact solution from a point that \
is not 0!'
                    )
                me = u_init
            else:
                t_init = 0.0

            def rhs(t, u):
                f = self.eval_f(u, t)
                return f.impl + f.expl  # evaluate only explicitly rather than IMEX

            tol = 100 * np.finfo(float).eps

            me[:] = solve_ivp(rhs, (t_init, t), me, rtol=tol, atol=tol).y[:, -1]

        return me
