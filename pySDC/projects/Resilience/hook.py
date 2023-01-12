from pySDC.core.Hooks import hooks


class log_error_estimates(hooks):
    """
    Record data required for analysis of problems in the resilience project
    """

    def pre_run(self, step, level_number):
        """
        Record los conditiones initiales
        """
        super(log_error_estimates, self).pre_run(step, level_number)
        L = step.levels[level_number]
        self.add_to_stats(process=0, time=0, level=0, iter=0, sweep=0, type='u0', value=L.u[0])

    def post_step(self, step, level_number):
        """
        Record final solutions as well as step size and error estimates
        """
        super(log_error_estimates, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='dt',
            value=L.dt,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_embedded',
            value=L.status.get('error_embedded_estimate'),
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_extrapolated',
            value=L.status.get('error_extrapolation_estimate'),
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restart',
            value=int(step.status.get('restart')),
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='sweeps',
            value=step.status.iter,
        )


class log_global_error(log_error_estimates):
    '''
    This class stores the "true" global error (may be approximated with scipy reference solution)
    '''

    def post_step(self, step, level_number):

        super(log_global_error, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_glob',
            value=abs(L.prob.u_exact(t=L.time + L.dt) - L.u[-1]),
        )


class log_local_error(log_error_estimates):
    '''
    This class stores the "true" local error (may be approximated with scipy reference solution)
    '''

    def post_step(self, step, level_number):

        super(log_local_error, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_loc',
            value=abs(L.prob.u_exact(t=L.time + L.dt, u_init=L.u[0], t_init=L.time) - L.u[-1]),
        )
