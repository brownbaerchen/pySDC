from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.helpers.problem_helper import get_finite_difference_matrix

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class HeatEquation(ptype):
    """
    This is a very basic implementation of a heat equation with finite differences and periodic boundaries.
    """

    # set datatype of solution and right hand side evaluations as class attributes
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, ndim=1, stencil_type='center', space_order=2, nu=1.0, freq=2):
        # create a tuple that can be used for pySDC datatype instantiation.
        # the format is (<shape>, <MPI communicator>, <datatype>)
        init = (nvars if ndim == 1 else [nvars for i in range(ndim)], None, np.dtype('float64'))

        # call the super init
        super().__init__(init=init)

        # setup 1D spatial grid
        dx = 1.0 / nvars
        xvalues = np.array([i * dx for i in range(nvars)])

        # build the finite difference matrix
        self.A = get_finite_difference_matrix(
            derivative=2,
            order=space_order,
            stencil_type=stencil_type,
            dx=dx,
            size=nvars,
            dim=ndim,
            bc='periodic',
        )
        self.A *= nu

        # make the attributes accessible outside this function
        self.xvalues = xvalues
        self.Id = sp.eye(np.power(nvars, ndim), format='csc')
        self._makeAttributeAndRegister(
            'nvars', 'stencil_type', 'space_order', 'ndim', 'nu', 'freq', localVars=locals(), readOnly=True
        )

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Parameters
        ----------
        u : dtype_u
            Current values.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            The RHS values.
        """
        f = self.f_init
        f[:] = self.A.dot(u.flatten()).reshape(self.init[0])
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        sol : dtype_u
            The solution of the linear solver.
        """
        sol = self.u_init
        sol[:] = spsolve(self.Id - factor * self.A, rhs.flatten()).reshape(self.init[0])
        return sol

    def u_exact(self, t, **kwargs):
        """
        Routine to compute the exact solution at time t. Notably this can be used for initial conditions at t=0.

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """
        ndim, freq, nu, dx, sol = self.ndim, self.freq, self.nu, self.dx, self.u_init

        if ndim == 1:
            x = self.grids
            rho = (2.0 - 2.0 * np.cos(np.pi * freq[0] * dx)) / dx**2
            if freq[0] > 0:
                sol[:] = np.sin(np.pi * freq[0] * x) * np.exp(-t * nu * rho)
        elif ndim == 2:
            rho = (2.0 - 2.0 * np.cos(np.pi * freq[0] * dx)) / dx**2 + (
                2.0 - 2.0 * np.cos(np.pi * freq[1] * dx)
            ) / dx**2
            x, y = self.grids
            sol[:] = np.sin(np.pi * freq[0] * x) * np.sin(np.pi * freq[1] * y) * np.exp(-t * nu * rho)
        elif ndim == 3:
            rho = (
                (2.0 - 2.0 * np.cos(np.pi * freq[0] * dx)) / dx**2
                + (2.0 - 2.0 * np.cos(np.pi * freq[1] * dx))
                + (2.0 - 2.0 * np.cos(np.pi * freq[2] * dx)) / dx**2
            )
            x, y, z = self.grids
            sol[:] = (
                np.sin(np.pi * freq[0] * x)
                * np.sin(np.pi * freq[1] * y)
                * np.sin(np.pi * freq[2] * z)
                * np.exp(-t * nu * rho)
            )

        return sol


##########################################################################################################
