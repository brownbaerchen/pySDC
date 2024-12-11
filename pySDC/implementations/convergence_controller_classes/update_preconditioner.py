from pySDC.core.convergence_controller import ConvergenceController, Pars, Status


class UpdatePreconditioner(ConvergenceController):
    """
    Update coefficients of the preconditioner before the iteration.
    """

    def pre_iteration_processing(self, controller, S, **kwargs):
        for lvl in S.levels:
            lvl.sweep.updateVariableCoeffs(S.status.iter)
