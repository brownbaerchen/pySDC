from pySDC.core.problem import Problem
from pySDC.helpers.spectral_helper import SpectralHelper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class GenericSpectralLinear(Problem):
    """
    Generic class to solve problems of the form M u_t + L u = y, with mass matrix M, linear operator L and some right hand side y using spectral methods.
    L may contain algebraic conditions, as long as (M + dt L) is invertible.

    """

    def __init__(self, bases, components, comm=None, *args, **kwargs):
        self.spectral = SpectralHelper(comm=comm)

        for base in bases:
            self.spectral.add_axis(**base)
        self.spectral.add_component(components)

        self.spectral.setup_fft()

        super().__init__(init=self.spectral.init)

    def __getattr__(self, name):
        return getattr(self.spectral, name)

    def _setup_operator(self, LHS):
        """
        Setup a sparse linear operator by adding relationships. See documentation for ``GenericSpectralLinear.setup_L`` to learn more.

        Args:
            LHS (dict): Equations to be added to the operator

        Returns:
            sparse linear operator
        """
        operator = self.get_empty_operator_matrix()
        for line, equation in LHS.items():
            self.add_equation_lhs(operator, line, equation)
        return self.convert_operator_matrix_to_operator(operator)

    def setup_L(self, LHS):
        """
        Setup the left hand side of the linear operator L and store it in ``self.L``.

        The argument is meant to be a dictionary with the line you want to write the equation in as the key and the relationship between components as another dictionary. For instance, you can add an algebraic condition capturing a first derivative relationship between u and ux as follows:

        ```
        Dx = self.get_self.get_differentiation_matrix(axes=(0,))
        I = self.get_Id()
        LHS = {'ux': {'u': Dx, 'ux': -I}}
        self.setup_L(LHS)
        ```

        If you put zero as right hand side for the solver in the line for ux, ux will contain the x-derivative of u afterwards.

        Args:
            LHS (dict): Dictionary containing the equations.
        """
        self.L = self._setup_operator(LHS)

    def setup_M(self, LHS):
        '''
        See documentation of ``GenericSpectralLinear.setup_L``.
        '''
        self.M = self._setup_operator(LHS)

    def solve_system(self, rhs, factor, *args, **kwargs):
        sol = self.u_init

        rhs_hat = self.spectral.transform(rhs)
        rhs_hat = (self.M @ rhs_hat.flatten()).reshape(sol.shape)

        rhs = self.spectral.itransform(rhs_hat)
        rhs = self.spectral.put_BCs_in_rhs(rhs)
        rhs_hat = self.spectral.transform(rhs)

        A = self.M + factor * self.L
        A = self.spectral.put_BCs_in_matrix(A)

        sol_hat = (self.spectral.sparse_lib.linalg.spsolve(A, rhs_hat.flatten())).reshape(sol.shape)

        sol[:] = self.spectral.itransform(
            sol_hat,
        )
        return sol
