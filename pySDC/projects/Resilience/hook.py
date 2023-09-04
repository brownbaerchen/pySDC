from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate
from pySDC.implementations.hooks.log_extrapolated_error_estimate import LogExtrapolationErrorEstimate
from pySDC.implementations.hooks.log_step_size import LogStepSize


hook_collection = [LogSolution, LogEmbeddedErrorEstimate, LogExtrapolationErrorEstimate, LogStepSize]


class LogData(hooks):
    """
    Record data required for analysis of problems in the resilience project
    """

    def pre_run(self, step, level_number):
        """
        Record initial conditions
        """
        super().pre_run(step, level_number)

        L = step.levels[level_number]
        self.add_to_stats(process=0, time=0, level=0, iter=0, sweep=0, type='u0', value=L.u[0])

    def post_step(self, step, level_number):
        """
        Record final solutions as well as step size and error estimates
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        # add the following with two names because I use both in different projects -.-
        self.increment_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='sweeps',
            value=step.status.iter,
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='k',
            value=step.status.iter,
        )


class LogUold(hooks):
    """
    Log last iterate at the end of the step. Since the hook comes after override of uold, we need to do this in each
    iteration. But we don't know which will be the last, so we just do `iter=-1` to override the previous value.
    """

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)
        L = step.levels[level_number]
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=-1,
            sweep=L.status.sweep,
            type='uold',
            value=L.uold[-1],
        )


class LogUAllIter(hooks):
    """
    Log solution and errors after each iteration
    """

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='error_embedded_estimate',
            value=L.status.get('error_embedded_estimate'),
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='error_extrapolation_estimate',
            value=L.status.get('error_extrapolation_estimate'),
        )


class LogU0Increment(hooks):
    """
    Log the increment in the initial conditions between iterations. I want to use this to analyse GSSDC a bit.
    """

    def __init__(self):
        super().__init__()
        self._u0_last = [None]

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)

        lvl = step.levels[level_number]

        if None not in self._u0_last:
            self.add_to_stats(
                process=step.status.slot,
                time=lvl.time,
                level=lvl.level_index,
                iter=step.status.iter,
                sweep=lvl.status.sweep,
                type='u0_increment',
                value=abs(lvl.u[0] - self._u0_last) / abs(lvl.u[0]),
            )

        self._u0_last = lvl.u[0] * 1.0

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self._u0_last = [None]
