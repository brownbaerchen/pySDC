import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

from mpi4py_fft import newDistArray
import scipy.sparse as sp
from scipy.linalg import inv


class BrusselatorBase(ptype):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, init=None, alpha=0.1, imex=True):
        L = 1.0

        self._makeAttributeAndRegister('alpha', 'L', 'imex', localVars=locals(), readOnly=True)

        self.iU = 0
        self.iV = 1

        super().__init__(init=init)

        if imex:
            type(self).dtype_f = imex_mesh
        else:
            self.work_counters['newton'] = WorkCounter()

        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        iU, iV = self.iU, self.iV
        x, y = self.X[0], self.X[1]

        f = self.dtype_f(self.init)

        if self.imex:
            # evaluate Laplacian to be solved implicitly
            f.impl[...] = self._eval_f_Laplacian(u, f.impl, t)

            # evaluate autonomous part
            f.expl[...] = self._eval_f_nonlin(u, f.expl, t)
        else:
            f[...] = self._eval_f_Laplacian(u, f, t) + self._eval_f_nonlin(u, f, t)

        self.work_counters['rhs']()
        return f

    def _eval_f_Laplacian(self, u, f, t):
        raise NotImplementedError()

    def _eval_f_nonlin(self, u, f, t):
        iU, iV = self.iU, self.iV

        f[iU, ...] = 1.0 + u[iU] ** 2 * u[iV] - 4.4 * u[iU]
        f[iV, ...] = 3.4 * u[iU] - u[iU] ** 2 * u[iV]

        # add time-dependent source term
        if t >= 1.1:
            x, y = self.X[0], self.X[1]
            mask = (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2
            f[iU][mask] += 5.0

        return f

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Initial conditions.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        iU, iV = self.iU, self.iV
        x, y = self.X[0], self.X[1]

        me = self.dtype_u(self.init, val=0.0)

        if t == 0:
            me[iU, ...] = 22.0 * y * (1 - y / self.L) ** (3.0 / 2.0) / self.L
            me[iV, ...] = 27.0 * x * (1 - x / self.L) ** (3.0 / 2.0) / self.L
        else:

            def eval_rhs(t, u):
                f = self.eval_f(u.reshape(self.init[0]), t)
                if self.imex:
                    return (f.impl + f.expl).flatten()
                else:
                    return f.flatten()

            me[...] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init, first_step=1e-9)

        return me

    def solve_system(self, *args, **kwargs):
        if self.imex:
            return self._solve_system_imex(*args, **kwargs)
        else:
            return self._solve_system_fully_implicit(*args, **kwargs)

    def _solve_system_fully_implicit(self, *args, **kwargs):
        raise NotImplementedError

    def _solve_system_imex(self, *args, **kwargs):
        raise NotImplementedError

    def get_fig(self):  # pragma: no cover
        """
        Get a figure suitable to plot the solution of this problem

        Returns
        -------
        self.fig : matplotlib.pyplot.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plt.rcParams['figure.constrained_layout.use'] = True
        self.fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=((8, 3)))
        divider = make_axes_locatable(axs[1])
        self.cax = divider.append_axes('right', size='3%', pad=0.03)
        return self.fig

    def plot(self, u, t=None, fig=None):  # pragma: no cover
        r"""
        Plot the solution. Please supply a figure with the same structure as returned by ``self.get_fig``.

        Parameters
        ----------
        u : dtype_u
            Solution to be plotted
        t : float
            Time to display at the top of the figure
        fig : matplotlib.pyplot.figure.Figure
            Figure with the correct structure

        Returns
        -------
        None
        """
        fig = self.get_fig() if fig is None else fig
        axs = fig.axes

        vmin = u.min()
        vmax = u.max()
        for i, label in zip([self.iU, self.iV], [r'$u$', r'$v$']):
            im = axs[i].pcolormesh(self.X[0], self.X[1], u[i], vmin=vmin, vmax=vmax)
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2e}')
        axs[0].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$y$')
        fig.colorbar(im, self.cax)


class Brusselator(BrusselatorBase):
    r"""
    Two-dimensional Brusselator from [1]_.
    This is a reaction-diffusion equation with non-autonomous source term:

    .. math::
        \frac{\partial u}{\partial t} = \varalpha \Delta u + 1 + u^2 v - 4.4u _ f(x,y,t),
        \frac{\partial v}{\partial t} = \varalpha \Delta v + 3.4u - u^2 v

    with the source term :math:`f(x,y,t) = 5` if :math:`(x-0.3)^2 + (y-0.6)^2 <= 0.1^2` and :math:`t >= 1.1` and 0 else.
    We discretize in a periodic domain of length 1 and solve with an IMEX scheme based on a spectral method for the
    Laplacian which we invert implicitly. We treat the reaction and source terms explicitly.

    References
    ----------
    .. [1] https://link.springer.com/book/10.1007/978-3-642-05221-7
    """

    def __init__(self, nvars=None, comm=MPI.COMM_WORLD, **kwargs):
        """Initialization routine"""
        nvars = (128,) * 2 if nvars is None else nvars

        if not (isinstance(nvars, tuple) and len(nvars) == 2):
            raise ProblemError('Need two dimensions')

        # Create FFT structure
        self.ndim = len(nvars)
        axes = tuple(range(self.ndim))
        self.fft = PFFT(
            comm,
            list(nvars),
            axes=axes,
            dtype=np.float64,
            collapse=True,
            backend='fftw',
        )

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, False)

        # prepare the array with two components
        shape = (2,) + tmp_u.shape

        super().__init__(init=(shape, comm, tmp_u.dtype), **kwargs)
        self._makeAttributeAndRegister('nvars', 'comm', localVars=locals(), readOnly=True)

        L = np.array([self.L] * self.ndim, dtype=float)

        # get local mesh for distributed FFT
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = X[i] * L[i] / N[i]
        self.X = [np.broadcast_to(x, self.fft.shape(False)) for x in X]

        # get local wavenumbers and Laplace operator
        s = self.fft.local_slice()
        N = self.fft.global_shape()
        k = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1.0 / N[-1]).astype(int))
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = np.meshgrid(*K, indexing='ij', sparse=True)
        Lp = 2 * np.pi / L
        for i in range(self.ndim):
            Ks[i] = (Ks[i] * Lp[i]).astype(float)
        K = [np.broadcast_to(k, self.fft.shape(True)) for k in Ks]
        K = np.array(K).astype(float)
        self.K2 = np.sum(K * K, 0, dtype=float)

        # Need this for diagnostics
        self.dx = self.L / nvars[0]
        self.dy = self.L / nvars[1]

    def _eval_f_Laplacian(self, u, f, t):
        for i in [self.iU, self.iV]:
            u_hat = self.fft.forward(u[i, ...])
            lap_u_hat = -self.alpha * self.K2 * u_hat
            f[i, ...] = self.fft.backward(lap_u_hat, f[i, ...])
        return f

    def _solve_system_imex(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            Solution.
        """
        assert self.imex

        me = self.dtype_u(self.init)

        for i in [self.iU, self.iV]:
            rhs_hat = self.fft.forward(rhs[i, ...])
            rhs_hat /= 1.0 + factor * self.K2 * self.alpha
            me[i, ...] = self.fft.backward(rhs_hat, me[i, ...])

        return me


class BrusselatorFD(BrusselatorBase):
    def __init__(
        self,
        nvars=None,
        order=8,
        stencil_type='center',
        lintol=1e-7,
        liniter=99,
        solver_type='cg',
        inexact_linear_ratio=None,
        newton_maxiter=99,
        newton_tol=1e-7,
        **kwargs,
    ):
        from pySDC.helpers import problem_helper
        import scipy.sparse as sp

        nvars = (128,) * 2 if nvars is None else nvars
        if not (isinstance(nvars, tuple) and len(nvars) == 2):
            raise ProblemError('Need two dimensions')

        shape = (2,) + nvars
        super().__init__(init=(shape, None, np.dtype('float64')), **kwargs)

        dx, xvalues = problem_helper.get_1d_grid(size=nvars[0], bc='periodic', left_boundary=0.0, right_boundary=1.0)

        self.A, _ = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=order,
            stencil_type=stencil_type,
            dx=dx,
            size=nvars[0],
            dim=2,
            bc='periodic',
        )
        self.A *= self.alpha

        self.xvalues = xvalues
        self.X = np.meshgrid(self.xvalues, self.xvalues)
        self.Id = sp.eye(np.prod(nvars), format='csc')

        # store attribute and register them as parameters
        self._makeAttributeAndRegister('nvars', 'stencil_type', 'order', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister(
            'lintol',
            'liniter',
            'solver_type',
            'inexact_linear_ratio',
            'newton_maxiter',
            'newton_tol',
            localVars=locals(),
        )

        if self.solver_type != 'direct':
            self.work_counters[self.solver_type] = WorkCounter()

        if self.solver_type == 'direct':
            from scipy.sparse.linalg import spsolve as solver
        elif self.solver_type == 'gmres':
            from scipy.sparse.linalg import gmres as solver
        elif self.solver_type == 'cg':
            from scipy.sparse.linalg import cg as solver
        else:
            raise NotImplementedError(f'Solver {self.solver_type!r} not implemented!')
        self.solver = solver

    def _eval_f_Laplacian(self, u, f, t):
        for i in [self.iU, self.iV]:
            f[i, ...] = self.A.dot(u[i].flatten()).reshape(self.nvars)
        return f

    def _solve_system_imex(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            Solution.
        """
        assert self.imex

        sol = self.dtype_u(self.init)

        solver_type, Id, A, nvars, lintol, liniter, sol = (
            self.solver_type,
            self.Id,
            self.A,
            self.nvars,
            self.lintol,
            self.liniter,
            self.u_init,
        )

        for i in [self.iU, self.iV]:
            if solver_type == 'direct':
                sol[i, ...] = self.solver(Id - factor * A, rhs[i, ...].flatten()).reshape(nvars)
            elif solver_type == 'gmres':
                sol[i, ...] = self.solver(
                    Id - factor * A,
                    rhs[i, ...].flatten(),
                    x0=u0[i, ...].flatten(),
                    tol=lintol,
                    maxiter=liniter,
                    atol=0,
                    callback=self.work_counters[self.solver_type],
                    callback_type='legacy',
                )[0].reshape(nvars)
            elif solver_type == 'cg':
                sol[i, ...] = self.solver(
                    Id - factor * A,
                    rhs[i, ...].flatten(),
                    x0=u0[i, ...].flatten(),
                    tol=lintol,
                    maxiter=liniter,
                    atol=0,
                    callback=self.work_counters[self.solver_type],
                )[0].reshape(nvars)
            else:
                raise NotImplementedError(f'solver type {solver_type!r} not implemented')

        return sol

    def get_non_linear_Jacobian(self, current_sol):
        dudu = sp.diags((2 * current_sol[self.iU] * current_sol[self.iV] - 4.4).flatten())
        dudv = sp.diags((current_sol[self.iU] ** 2).flatten())
        dvdu = sp.diags((3.4 - 2 * current_sol[self.iU] * current_sol[self.iV]).flatten())
        dvdv = sp.diags((-current_sol[self.iU] ** 2).flatten())
        return sp.block_array([[dudu, dudv], [dvdu, dvdv]])

    def _solve_system_fully_implicit(self, rhs, factor, u0, t):
        r"""
        Simple Newton solver for :math:`(I - factor \cdot f)(\vec{u}) = \vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """
        u = self.dtype_u(u0)
        res = np.inf
        delta = self.dtype_u(self.init, val=0.0)

        Id = sp.eye(m=np.prod(u.shape))
        A = sp.block_diag(mats=(self.A, self.A))
        zero = np.zeros_like(u.flatten())

        for n in range(0, self.newton_maxiter):
            # assemble G such that G(u) = 0 at the solution of the step
            G = (u - factor * self.eval_f(u, t) - rhs).flatten()
            self.work_counters[
                'rhs'
            ].decrement()  # Work regarding construction of the Jacobian etc. should count into the Newton iterations only

            res = np.linalg.norm(G, np.inf)
            if res <= self.newton_tol and n > 0:  # we want to make at least one Newton iteration
                break

            if self.inexact_linear_ratio:
                self.lintol = max([res * self.inexact_linear_ratio, self.min_lintol])

            # assemble Jacobian J of G
            J = Id - factor * (A + self.get_non_linear_Jacobian(u))

            # solve the linear system
            if self.solver_type == 'direct':
                delta = self.solver(J, G)
            elif self.solver_type == 'gmres':
                delta, info = self.solver(
                    J,
                    G.flatten(),
                    x0=zero,
                    tol=self.lintol,
                    maxiter=self.liniter,
                    atol=0,
                    callback=self.work_counters[self.solver_type],
                )
            elif self.solver_type == 'cg':
                delta = self.solver(
                    J,
                    G.flatten(),
                    x0=zero,
                    tol=self.lintol,
                    maxiter=self.liniter,
                    atol=0,
                    callback=self.work_counters[self.solver_type],
                )[0]
            else:
                raise NotImplementedError(f'solver type {solver_type!r} not implemented')

            if not np.isfinite(delta).all():
                break

            # update solution
            u = u - delta.reshape(u.shape)

            self.work_counters['newton']()

        return u

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Initial conditions.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """
        if self.imex or t == 0:
            return super().u_exact(t, u_init, t_init)

        me = self.dtype_u(self.init, val=0.0)
        A = sp.block_diag(mats=(self.A, self.A))

        def jac(t, u):
            return A + self.get_non_linear_Jacobian(u.reshape(me.shape))

        def eval_rhs(t, u):
            print(t)
            f = self.eval_f(u.reshape(self.init[0]), t)
            return f.flatten()

        me[...] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init, jac=jac, method='BDF')

        return me


# class BrusselatorFullyImplicit(Brusselator):
#     dtype_u = mesh
#     dtype_f = mesh
#
#     def __init__(self, newton_maxiter, newton_tol, **kwargs):
#         super().__init__(**kwargs)
#         self._makeAttributeAndRegister('newton_maxiter', 'newton_tol', localVars=locals(), readOnly=False)
#
#     def eval_f(self, u, t):
#         """
#         Routine to evaluate the right-hand side of the problem.
#
#         Parameters
#         ----------
#         u : dtype_u
#             Current values of the numerical solution.
#         t : float
#             Current time of the numerical solution is computed.
#
#         Returns
#         -------
#         f : dtype_f
#             The right-hand side of the problem.
#         """
#         iU, iV = self.iU, self.iV
#         x, y = self.X[0], self.X[1]
#
#         f = self.dtype_f(self.init)
#
#         # evaluate Laplacian
#         for i in [self.iU, self.iV]:
#             u_hat = self.fft.forward(u[i, ...])
#             lap_u_hat = -self.alpha * self.K2 * u_hat
#             f[i, ...] += self.fft.backward(lap_u_hat)
#
#         # evaluate autonomous part
#         f[iU, ...] += 1.0 + u[iU] ** 2 * u[iV] - 4.4 * u[iU]
#         f[iV, ...] += 3.4 * u[iU] - u[iU] ** 2 * u[iV]
#
#         # add non-autonomous part
#         if t >= 1.1:
#             mask = (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2
#             f[iU][mask] += 5.0
#
#         return f
#
#     def solve_system(self, rhs, factor, u0, t):
#         """
#         Newton solver
#
#         Parameters
#         ----------
#         rhs : dtype_f
#             Right-hand side for the linear system.
#         factor : float
#             Abbrev. for the node-to-node stepsize (or any other factor required).
#         u0 : dtype_u
#             Initial guess for the iterative solver (not used here so far).
#         t : float
#             Current time (e.g. for time-dependent BCs).
#
#         Returns
#         -------
#         me : dtype_u
#             Solution.
#         """
#         iU, iV = self.iU, self.iV
#         me = self.dtype_u(self.init)
#         u, v = me[self.iU], me[self.iV]
#
#         # for i in [self.iU, self.iV]:
#         #     rhs_hat = self.fft.forward(rhs[i, ...])
#         #     rhs_hat /= 1.0 + factor * self.K2 * self.alpha
#         #     me[i, ...] = self.fft.backward(rhs_hat, me[i, ...])
#
#         # start Newton iteration
#         n = 0
#         res = 99
#         while n < self.newton_maxiter:
#             # assemble G such that G(u) = 0 at the solution of the step
#             source = np.ones_like(u)
#             if t >= 1.1:
#                 mask = (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2
#                 source[mask] += 5.0
#
#             u_hat = self.fft.forward(u)
#             v_hat = self.fft.forward(v)
#             u2v_hat = self.fft.forward(u*u*v)
#             source_hat = self.fft.forward(source)
#
#             G = self.dtype_u(self.init)
#             G[iU] = u_hat - factor * (u2v_hat - 4.4*u_hat + self.alpha*self.K2 * u_hat + source_hat) - rhs
#             G[iV] = v_hat - factor * (-u2v_hat + 3.4*u_hat + self.alpha*self.K2 * v_hat)
#
#             res = np.linalg.norm(G, np.inf)
#             if res < self.newton_tol or np.isnan(res):
#                 break
#
#             # prefactor for dg/du
#             c = 1.0 / (-2 * dt**2 * mu * x1 * x2 - dt**2 - 1 + dt * mu * (1 - x1**2))
#             # assemble dg/du
#             dg = c * np.array([[dt * mu * (1 - x1**2) - 1, -dt], [2 * dt * mu * x1 * x2 + dt, -1]])
#
#             # newton update: u1 = u0 - g/dg
#             u -= np.dot(dg, g)
#
#             # set new values and increase iteration count
#             x1 = u[0]
#             x2 = u[1]
#             n += 1
#             self.work_counters['newton']()
#
#         if np.isnan(res) and self.stop_at_nan:
#             self.logger.warning('Newton got nan after %i iterations...' % n)
#             raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
#         elif np.isnan(res):
#             self.logger.warning('Newton got nan after %i iterations...' % n)
#
#         if n == self.newton_maxiter and self.crash_at_maxiter:
#             raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))
#
#         return u
#
#         return me
