import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController


class StepSizeLimiter(ConvergenceController):
    """
    Class to set limits to adaptive step size computation during run time

    Please supply dt_min or dt_max in the params to limit in either direction
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": +92,
            "dt_min": 0,
            "dt_max": np.inf,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def dependencies(self, controller, description, **kwargs):
        """
        Load the slope limiter if needed.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        slope_limiter_keys = ['dt_slope_min', 'dt_slope_max']
        available_keys = [me for me in slope_limiter_keys if me in self.params.__dict__.keys()]

        if len(available_keys) > 0:
            slope_limiter_params = {key: self.params.__dict__[key] for key in available_keys}
            slope_limiter_params['control_order'] = self.params.control_order - 1
            controller.add_convergence_controller(
                StepSizeSlopeLimiter, params=slope_limiter_params, description=description
            )

        return None

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Enforce an upper and lower limit to the step size here.
        Be aware that this is only tested when a new step size has been determined. That means if you set an initial
        value for the step size outside of the limits, and you don't do any further step size control, that value will
        go through.
        Also, the final step is adjusted such that we reach Tend as best as possible, which might give step sizes below
        the lower limit set here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        for L in S.levels:
            if L.status.dt_new is not None:
                if L.status.dt_new < self.params.dt_min:
                    self.log(
                        f"Step size is below minimum, increasing from {L.status.dt_new:.2e} to \
{self.params.dt_min:.2e}",
                        S,
                    )
                    L.status.dt_new = self.params.dt_min
                elif L.status.dt_new > self.params.dt_max:
                    self.log(
                        f"Step size exceeds maximum, decreasing from {L.status.dt_new:.2e} to {self.params.dt_max:.2e}",
                        S,
                    )
                    L.status.dt_new = self.params.dt_max

        return None


class StepSizeSlopeLimiter(ConvergenceController):
    """
    Class to set limits to adaptive step size computation during run time

    Please supply dt_min or dt_max in the params to limit in either direction
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": 91,
            "dt_slope_min": 0,
            "dt_slope_max": np.inf,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Enforce an upper and lower limit to the slope of the step size here.
        The final step is adjusted such that we reach Tend as best as possible, which might give step sizes below
        the lower limit set here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        for L in S.levels:
            if L.status.dt_new is not None:
                if L.status.dt_new / L.params.dt < self.params.dt_slope_min:
                    dt_new = L.params.dt * self.params.dt_slope_min
                    self.log(
                        f"Step size slope is below minimum, increasing from {L.status.dt_new:.2e} to \
{dt_new:.2e}",
                        S,
                    )
                    L.status.dt_new = dt_new
                elif L.status.dt_new / L.params.dt > self.params.dt_slope_max:
                    dt_new = L.params.dt * self.params.dt_slope_max
                    self.log(
                        f"Step size slope exceeds maximum, decreasing from {L.status.dt_new:.2e} to \
{dt_new:.2e}",
                        S,
                    )
                    L.status.dt_new = dt_new

        return None


class WorkLimiter(ConvergenceController):
    def __init__(self, controller, params, description, **kwargs):
        self.work = [None] * 2
        self.dt = [None] * 2
        super().__init__(controller, params, description, **kwargs)

    def setup(self, controller, params, description, **kwargs):
        defaults = {
            "control_order": +210,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def get_new_step_size(self, controller, S, **kwargs):
        L = S.levels[0]
        if S.status.done and not S.status.restart:
            self.work[:-1] = self.work[1:]
            self.work[-1] = S.status.iter * 1.0
            self.dt[:-1] = self.dt[1:]
            self.dt[-1] = S.dt * 1.0
            # print(self.work, self.dt)

        if None not in self.work and L.status.dt_new is not None:
            # first order finite difference approximation of the derivative of the "power" wrt step size
            diff = self.work[-1] / self.dt[-1] - self.work[-2] / self.dt[-2]  # / (self.dt[-1] - self.dt[-2])
            if diff > 0 and L.status.dt_new > L.params.dt and self.dt[-1] > self.dt[-2]:
                self.log(f'Found decrease in power by {diff:.2e}! Prohibiting step size increase!', S)
                L.status.dt_new = L.params.dt
        # if S.status.done:
        #    print(self.work, self.dt, S.status.done, S.status.restart, S.status.iter, S.dt)
        #    breakpoint()
