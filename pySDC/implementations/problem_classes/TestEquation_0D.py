from pySDC.core.Problem import ptype, WorkCounter


# noinspection PyUnusedLocal
class testequation0dXPU(ptype):
    r"""
    This class implements the simple test equation of the form

    .. math::
        \frac{d u(t)}{dt} = A u(t)

    for :math:`A = diag(\lambda_1, .. ,\lambda_n)`.

    It is compatible with both CPU and GPU, but requires setting class attributes to select the corresponding numerical
    library. Use the classmethod `get_XPU_version` to get a runnable problem class.

    Parameters
    ----------
    lambdas : sequence of array_like, optional
        List of lambda parameters.
    u0 : sequence of array_like, optional
        Initial condition.

    Attributes
    ----------
    A : scipy.sparse.csc_matrix
        Diagonal matrix containing :math:`\lambda_1,..,\lambda_n`.
    """

    @classmethod
    def get_XPU_version(cls, version):
        """
        Get a runnable version for either CPU or GPU by specifying this as `version`.

        Parameters
        ----------
        version : str
            Supply "GPU" or "CPU" to obtain the desired implementation

        Returns
        -------
        pySDC.Problem
            A problem class implementing the desired implementation
        """
        if version == 'CPU':
            return testequation0d
        elif version == 'GPU':
            return testequation0dGPU
        else:
            from pySDC.core.Errors import ParameterError

            raise ParameterError(f'Don\'t know version {version}! Please choose \'CPU\' or \'GPU\'!')

    def __init__(self, lambdas=None, u0=0.0):
        """Initialization routine"""
        if lambdas is None:
            re = self.xp.linspace(-30, 19, 50)
            im = self.xp.linspace(-50, 49, 50)
            lambdas = self.xp.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
                (len(re) * len(im))
            )

        assert not any(isinstance(i, list) for i in lambdas), 'ERROR: expect flat list here, got %s' % lambdas
        nvars = len(lambdas)
        assert nvars > 0, 'ERROR: expect at least one lambda parameter here'

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, self.xp.dtype('complex128')))

        lambdas = self.xp.array(lambdas)
        self.A = self.__get_A(lambdas, self.xsp)
        self._makeAttributeAndRegister('nvars', 'lambdas', 'u0', localVars=locals(), readOnly=True)
        self.work_counters['rhs'] = WorkCounter()

    @staticmethod
    def __get_A(lambdas, xsp):
        """
        Helper function to assemble FD matrix A in sparse format.

        Parameters
        ----------
        lambdas : sequence of array_like
            List of lambda parameters.

        Returns
        -------
        scipy.sparse.csc_matrix
            Diagonal matrix A in CSC format.
        """

        A = xsp.diags(lambdas)
        return A

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

        f = self.dtype_f(self.init)
        f[:] = self.A.dot(u)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

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
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)
        L = self.splu(self.xsp.eye(self.nvars, format='csc') - factor * self.A)
        me[:] = L.solve(rhs)
        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : pySDC.problem.testequation0d.dtype_u
            Initial solution.
        t_init : float
            The initial time.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        u_init = (self.u0 if u_init is None else u_init) * 1.0
        t_init = 0.0 if t_init is None else t_init * 1.0

        me = self.dtype_u(self.init)
        me[:] = u_init * self.xp.exp((t - t_init) * self.lambdas)
        return me


class testequation0d(testequation0dXPU):
    """
    CPU implementation of `testequation0dXPU`
    """

    from pySDC.implementations.datatype_classes.mesh import mesh
    import numpy as xp
    import scipy.sparse as xsp
    from scipy.sparse.linalg import splu as _splu

    dtype_u = mesh
    dtype_f = mesh
    splu = staticmethod(_splu)


class testequation0dGPU(testequation0dXPU):
    """
    GPU implementation of `testequation0dXPU`
    """

    from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh
    import cupy as xp
    import cupyx.scipy.sparse as xsp
    from cupyx.scipy.sparse.linalg import splu as _splu

    dtype_u = cupy_mesh
    dtype_f = cupy_mesh
    splu = staticmethod(_splu)
