from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers.spectral_helper import SpectralHelper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class GenericSpectralLinear(Problem):
    """
    Generic class to solve problems of the form M u_t + L u = y, with mass matrix M, linear operator L and some right hand side y using spectral methods.
    L may contain algebraic conditions, as long as (M + dt L) is invertible.

    """

    def __init__(
        self,
        bases,
        components,
        comm=None,
        preconditioning='D2T',
        solver_type='direct',
        solver_args=None,
        *args,
        **kwargs,
    ):
        self.spectral = SpectralHelper(comm=comm)

        for base in bases:
            self.spectral.add_axis(**base)
        self.spectral.add_component(components)

        self.spectral.setup_fft()

        super().__init__(init=self.spectral.init)

        self.solver_type = solver_type
        self.solver_args = {} if solver_args is None else solver_args

        self.work_counters[solver_type] = WorkCounter()

        self.setup_right_preconditioner(preconditioning)

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
        Setup mass matrix, see documentation of ``GenericSpectralLinear.setup_L``.
        '''
        self.M = self._setup_operator(LHS)

    def setup_right_preconditioner(self, preconditioning='D2T'):
        basis_conversion_matrix = self.spectral.get_basis_change_matrix(direction=preconditioning)
        Pr_lhs = {comp: {comp: basis_conversion_matrix} for comp in self.components}
        self.Pr = self._setup_operator(Pr_lhs)

    def solve_system(self, rhs, dt, u0=None, **kwargs):
        """
        Solve (M + dt*L)u=rhs. This requires that you setup the operators before using the functions ``GenericSpectralLinear.setup_L`` and ``GenericSpectralLinear.setup_M``. Note that the mass matrix need not be invertible, as long as (M + dt*L) is. This allows to solve some differential algebraic equations.

        Note that in implicit Euler, the right hand side will be composed of the initial conditions. We don't want that in lines that don't depend on time. Therefore, we multiply the right hand side by the mass matrix. This means you can only do algebraic conditions that add up to zero. But you can easily overload this function with something more generic if needed.

        We use a tau method to enforce boundary conditions in Chebychov methods. This means we replace a line in the system matrix by the polynomials evaluated at a boundary and put the value we want there in the rhs at the respective position. Since we have to do that in spectral space along only the axis we want to apply the boundary condition to, we transform back to real space after applying the mass matrix, and then transform only along one axis, apply the boundary conditions and transform back. Then we transform along all dimensions again. If you desire speed, you may wish to overload this function with something less generic that avoids a few transformations.
        """
        sp = self.spectral.sparse_lib

        sol = self.u_init

        rhs_hat = self.spectral.transform(rhs)
        rhs_hat = (self.M @ rhs_hat.flatten()).reshape(rhs_hat.shape)
        rhs = self.spectral.itransform(rhs_hat)

        rhs = self.spectral.put_BCs_in_rhs(rhs) / dt
        rhs_hat = self.spectral.transform(rhs).flatten()

        A = self.M + dt * self.L
        A = self.spectral.put_BCs_in_matrix(A) / dt @ self.Pr

        if self.solver_type == 'direct':
            sol_hat = sp.linalg.spsolve(A, rhs_hat)
        elif self.solver_type == 'gmres':
            sol_hat, _ = sp.linalg.gmres(
                A,
                rhs_hat,
                x0=u0,
                **self.solver_args,
                callback=self.work_counters[self.solver_type],
                callback_type='legacy',
            )
        elif self.solver_type == 'cg':
            sol_hat, _ = sp.linalg.cg(
                A, rhs_hat, x0=u0, **self.solver_args, callback=self.work_counters[self.solver_type]
            )
        else:
            raise NotImplementedError(f'Solver {self.solver_type:!} not implemented in {type(self).__name__}!')

        sol_hat = self.Pr @ sol_hat
        sol[:] = self.spectral.itransform(sol_hat.reshape(sol.shape))
        return sol
