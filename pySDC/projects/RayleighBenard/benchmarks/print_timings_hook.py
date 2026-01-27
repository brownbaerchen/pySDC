from pySDC.implementations.hooks.log_timings import CPUTimings  # , GPUTimings


class PrintCPUTimings(CPUTimings):
    def post_step(self, *args, **kwargs):
        super().post_step(*args, **kwargs)
        t0 = self._Timings__t0_step
        t1 = self._Timings__t1_step
        self.logger.log(level=50, msg=f'CPU timing step: {t1-t0:.8f}s')
