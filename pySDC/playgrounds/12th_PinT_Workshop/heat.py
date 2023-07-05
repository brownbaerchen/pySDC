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
        self.dx = dx
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
            x = self.xvalues
            rho = (2.0 - 2.0 * np.cos(np.pi * freq * dx)) / dx**2
            if freq > 0:
                sol[:] = np.sin(np.pi * freq * x) * np.exp(-t * nu * rho)
        elif ndim == 2:
            rho = (2.0 - 2.0 * np.cos(np.pi * freq * dx)) / dx**2 + (2.0 - 2.0 * np.cos(np.pi * freq * dx)) / dx**2
            x, y = self.xvalues[None, :], self.xvalues[:, None]
            sol[:] = np.sin(np.pi * freq * x) * np.sin(np.pi * freq * y) * np.exp(-t * nu * rho)

        return sol


##########################################################################################################
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.hooks.log_solution import LogSolution

level_params = {}
level_params['dt'] = 0.1
level_params['restol'] = -1

step_params = {}
step_params['maxiter'] = 5

sweeper_params = {}
sweeper_params['QI'] = 'LU'
sweeper_params['num_nodes'] = 3
sweeper_params['quad_type'] = 'RADAU-RIGHT'
sweeper_params['initial_guess'] = 'spread'

problem_params = {}
problem_params['nvars'] = 32
problem_params['ndim'] = 2
problem_params['stencil_type'] = 'center'
problem_params['space_order'] = 4
problem_params['nu'] = 2e-1
problem_params['freq'] = 2

controller_params = {}
controller_params['logger_level'] = 20
controller_params['hook_class'] = LogSolution

description = {}
description['level_params'] = level_params
description['step_params'] = step_params
description['sweeper_params'] = sweeper_params
description['sweeper_class'] = generic_implicit
description['problem_params'] = problem_params
description['problem_class'] = HeatEquation

controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

prob = controller.MS[0].levels[0].prob
u_init = prob.u_exact(t=0)

u_end, stats = controller.run(u0=u_init, t0=0, Tend=0.3)
u = get_sorted(stats, type='u')

# plot the results
if problem_params['ndim'] == 1:
    ax = plt.subplot()
    x = prob.xvalues
    ax.plot(x, u_init, label=r'$u_0$', ls='--')

    for me in u:
        ax.plot(x, me[1], label=rf'$u(t={{{me[0]:.1f}}})$')
    ax.legend(frameon=False)
    ax.set_xlabel(r'$x$')
elif problem_params['ndim'] == 2:
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(u_init, vmin=0.0, vmax=0.8)

    for i in range(3):
        axs.flatten()[i + 1].set_title(fr'$t={{{u[i][0]:.1f}}}$')
        axs.flatten()[i + 1].imshow(u[i][1], vmin=0.0, vmax=0.8)

    fig.tight_layout()

plt.show()
