from pySDC.core.Hooks import hooks


class LogRestarts(hooks):
    """
    Record restarts as `restart` at the beginning of the step.
    """

    def __init__(self):
        super().__init__()
        self._total_num_restarts = 0
        self._total_num_steps = 0

    def post_step(self, step, level_number):
        """
        Record here if the step was restarted.

        Args:
            step (pySDC.Step.step): Current step
            level_number (int): Current level
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='restart',
            value=int(step.status.get('restart')),
        )
        self._total_num_restarts += int(step.status.get('restart'))
        self._total_num_steps += 1

    def post_run(self, step, level_number):
        super().post_run(step, level_number)
        self.logger.info(f'Restarted {self._total_num_restarts} times in total.')
        print(f'Restarted {self._total_num_restarts} times in total out of {self._total_num_steps} steps.')
