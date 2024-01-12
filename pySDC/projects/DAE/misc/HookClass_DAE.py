from pySDC.core.Hooks import hooks


class error_hook(hooks):
    """
    Hook class to log the error to the output generated by the sweeper after
    each time step.
    """

    def __init__(self):
        """Initialization routine"""
        super().__init__()

    def post_step(self, step, level_number):
        r"""
        Default routine called after each step.

        Parameters
        ----------
        step : pySDC.core.Step
            Current step.
        level_number : pySDC.core.level
            Current level number.
        """

        super().post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        L.sweep.compute_end_point()

        # compute and save errors
        # Note that the component from which the error is measured is specified here
        upde = P.u_exact(step.time + step.dt)
        err = abs(upde.diff - L.uend.diff)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='error_post_step',
            value=err,
        )


class LogGlobalErrorPostStepAlgebraicVariable(hooks):
    """
    Logs the global error in the algebraic variable
    """

    def post_step(self, step, level_number):
        r"""
        Default routine called after each step.

        Parameters
        ----------
        step : pySDC.core.Step
            Current step.
        level_number : pySDC.core.level
            Current level number.
        """

        super().post_step(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        e_global_algebraic = abs(upde.alg - L.uend.alg)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_global_algebraic_post_step',
            value=e_global_algebraic,
        )
