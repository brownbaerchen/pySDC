from pySDC.core.ConvergenceController import ConvergenceController


class PredictorCorrector(ConvergenceController):
    """
    Allow to change the preconditioner between iterations. This module is called predictor corrector because it can be
    used to perform a predictor sweep with a preconditioner like implicit Euler, which provides convergence even if the
    initial guess is bad and subsequent sweeps can be performed with something like LU, which often provides faster
    convergence if the initial guess is good.
    However, the module is more general, as you can supply a function that takes as input the iteration number as
    returns the preconditioner you want to use in that iteration.

    Please supply a function like the following which takes arbitrary keyword arguments as returns a string:
    ```
    def select_preconditioner_function(iter, **kwargs):
        if iter == 1:
            return 'IE'
        else:
            return 'LU'
    ```
    All information of the step status is available for applying conditions to, as well as the level number.
    """

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check if a suitable function is supplied in the parameters.
        """
        if 'select_preconditioner_function' not in params.keys():
            return (
                False,
                "Please supply a function `select_preconditioner_function` that selects preconditioners based on iteration information to the params!",
            )
        else:
            try:
                test_preconditioner = params['select_preconditioner_function'](iter=0, level=0, slot=0)
            except TypeError:
                return (
                    False,
                    'Please add `**kwargs` to the signature of `select_preconditioner_function` as this is required for abstraction!',
                )

            if type(test_preconditioner) is not str:
                return (
                    False,
                    '`select_preconditioner_function` must return a name of a preconditioner in string format for the sweeper to process the input!',
                )
        return True, ""

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here. The only relevant parameter here is the control order.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        return {"control_order": -200, **params}

    def pre_iteration_processing(self, controller, S, **kwargs):
        """
        Change the preconditioner according to the supplied rule.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The step

        Returns:
            None
        """
        for i in range(len(S.levels)):
            sweep = S.levels[i].sweep

            old_QI = f'{sweep.params.QI}'
            sweep.params.QI = self.params.select_preconditioner_function(**S.status.__dict__, level=i)
            sweep.QI = sweep.get_Qdelta_implicit(sweep.coll, qd_type=sweep.params.QI)

            if old_QI != sweep.params.QI:
                self.log(f'Changed the preconditioner on level {i} to {S.levels[i].sweep.params.QI}', S)
