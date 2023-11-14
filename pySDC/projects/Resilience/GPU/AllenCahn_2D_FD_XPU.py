import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh, comp2_mesh


# noinspection PyUnusedLocal
class allencahn_fullyimplicit_XPU(ptype):
    r"""
    Example implementing the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    for constant parameter :math:`\nu`. Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated *fully-implicitly*, i.e., the nonlinear system is solved by Newton.

    Parameters
    ----------
    nvars : tuple of int, optional
        Number of unknowns in the problem, e.g. ``nvars=(128, 128)``.
    nu : float, optional
        Problem parameter :math:`\nu`.
    eps : float, optional
        Scaling parameter :math:`\varepsilon`.
    newton_maxiter : int, optional
        Maximum number of iterations for the Newton solver.
    newton_tol : float, optional
        Tolerance for Newton's method to terminate.
    lin_tol : float, optional
        Tolerance for linear solver to terminate.
    lin_maxiter : int, optional
        Maximum number of iterations for the linear solver.
    radius : float, optional
        Radius of the circles.
    order : int, optional
        Order of the finite difference matrix.

    Attributes
    ----------
    A : scipy.spdiags
        Second-order FD discretization of the 2D laplace operator.
    dx : float
        Distance between two spatial nodes (same for both directions).
    xvalues : np.1darray
        Spatial grid points, here both dimensions have the same grid points.
    newton_itercount : int
        Number of iterations of Newton solver.
    lin_itercount
        Number of iterations of linear solver.
    newton_ncalls : int
        Number of calls of Newton solver.
    lin_ncalls : int
        Number of calls of linear solver.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=(128, 128),
        nu=2,
        eps=0.04,
        newton_maxiter=200,
        newton_tol=1e-12,
        lin_tol=1e-8,
        lin_maxiter=100,
        inexact_linear_ratio=None,
        radius=0.25,
        order=2,
        useGPU=False,
    ):
        """Initialization routine"""
        if useGPU:
            import cupy as cp
            import cupyx.scipy.sparse as csp
            from cupyx.scipy.sparse.linalg import cg
            from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh
            self.xp = cp
            self.xsp = csp
            self.cg = cg
            self.dtype_u = cupy_mesh
            self.dtype_f = cupy_mesh
        else:
            import scipy.sparse as sp
            from scipy.sparse.linalg import cg
            self.xp = np
            self.xsp = sp
            self.cg = cg

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(nvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % nvars)
        if nvars[0] != nvars[1]:
            raise ProblemError('need a square domain, got %s' % nvars)
        if nvars[0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, self.xp.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'nu',
            'eps',
            'radius',
            'order',
            'useGPU',
            localVars=locals(),
            readOnly=True,
        )
        self._makeAttributeAndRegister(
            'newton_maxiter',
            'newton_tol',
            'lin_tol',
            'lin_maxiter',
            'inexact_linear_ratio',
            localVars=locals(),
            readOnly=False,
        )

        # compute dx and get discretization matrix A
        self.dx = 1.0 / self.nvars[0]
        self.A, _ = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=self.order,
            stencil_type='center',
            dx=self.dx,
            size=self.nvars[0],
            dim=2,
            bc='periodic',
            useGPU=useGPU,
        )
        self.xvalues = self.xp.array([i * self.dx - 0.5 for i in range(self.nvars[0])])

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['linear'] = WorkCounter()


    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0).flatten()
        z = self.dtype_u(self.init, val=0.0).flatten()
        nu = self.nu
        eps2 = self.eps**2

        Id = self.xsp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) + 1.0 / eps2 * u * (1.0 - u**nu)) - rhs.flatten()

            # if g is close to 0, then we are done
            res = self.xp.linalg.norm(g, self.xp.inf)

            # do inexactness in the linear solver
            if self.inexact_linear_ratio:
                self.lin_tol = res * self.inexact_linear_ratio

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + 1.0 / eps2 * self.xsp.diags((1.0 - (nu + 1) * u**nu), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= self.cg(
                dg, g, x0=z, tol=self.lin_tol, maxiter=self.lin_maxiter, atol=0, callback=self.work_counters['linear']
            )[0]
            # increase iteration count
            n += 1
            # print(n, res)

            self.work_counters['newton']()

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1
        self.newton_itercount += n

        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f[:] = (self.A.dot(v) + 1.0 / self.eps**2 * v * (1.0 - v**self.nu)).reshape(self.nvars)

        self.work_counters['rhs']()
        return f

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        me = self.dtype_u(self.init, val=0.0)
        if t > 0:
            def eval_rhs(t, u):
                return self.eval_f(u.reshape(self.init[0]), t).flatten()

            if self.useGPU:
                prob = type(self)(**{**self.params, 'useGPU': False})
                me[:] = self.xp.asarray(prob.u_exact(t, u_init, t_init))
            else:
                me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)

        else:
            for i in range(self.nvars[0]):
                for j in range(self.nvars[1]):
                    r2 = self.xvalues[i] ** 2 + self.xvalues[j] ** 2
                    me[i, j] = self.xp.tanh((self.radius - self.xp.sqrt(r2)) / (self.xp.sqrt(2) * self.eps))

        return me
