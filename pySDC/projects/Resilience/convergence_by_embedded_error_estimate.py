import numpy as np
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence

class CheckConvergenceEmbeddedErrorEstimate(CheckConvergence):
    """
    Check the convergence based on an embedded error estimate.

    Checks for reaching a maximum number of iterations and residual tolerance are still performed.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": +201,
            "e_tol": np.inf,
        }
        return {**defaults, **params}

    def dependencies(self, controller, description, **kwargs):
        """
        Load dependencies on other convergence controllers here.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

        super().dependencies(controller, description, **kwargs)

        controller.add_convergence_controller(
            EstimateEmbeddedError.get_implementation("nonMPI" if type(controller) == controller_nonMPI else "MPI"),
            description=description,
        )

    def check_convergence(self, S):
        # check regular criteria
        converged = super().check_convergence(S)
        L = S.levels[0]
        error_estimate = L.status.error_embedded_estimate if L.status.error_embedded_estimate is not None else np.inf
        return (error_estimate < self.params.e_tol or converged) and not S.status.force_continue
