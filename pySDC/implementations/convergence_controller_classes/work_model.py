from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.implementations.hooks.log_work import LogWork
import numpy as np
from pySDC.helpers.problem_helper import get_finite_difference_stencil


class WorkModel(ConvergenceController):
    # TODO: docs
    def setup_status_variables(self, controller, **kwargs):
        # TODO: docs
        self.status = Status(['work_counters_last', 'work_counters'] + kwargs.get('status_variables', []))
        self.status.work_counters_last = {}
        self.status.work_counters = {}

    def record_work(self, S):
        # TODO: docs
        L = S.levels[0]
        P = L.prob

        for key in P.work_counters.keys():
            self.status.work_counters[key] = P.work_counters[key].niter - self.status.work_counters_last.get(key, 0)
            self.status.work_counters_last[key] = P.work_counters[key].niter


class ReduceWorkInAdaptivity(WorkModel):
    # TODO: docs

    def setup(self, controller, params, description, **kwargs):
        defaults = {
            'control_order': -40,
            'work_keys': [],
            'gamma': 0.5,
            'stencil_length': 2,
        }
        params = {**defaults, **super().setup(controller, params, description, **kwargs)}
        assert (
            type(params['work_keys']) == list
        ), f"Please supply a list for \"work_keys\" to the parameters for the {self.__class__.__name__} convergence controller! Got \"{params['work_keys']}\""

        assert (
            len(params['work_keys']) > 0
        ), f"Please supply some \"work_keys\" to the parameters for the {self.__class__.__name__} convergence controller! Got \"{params['work_keys']}\""
        assert (
            params['stencil_length'] >= 2
        ), f'Need to record the work of at least the last two steps, got only {params["stencil_length"]}!'
        return params

    def setup_status_variables(self, controller, **kwargs):
        # TODO: docs
        super().setup_status_variables(controller, status_variables=['work_last_step', 'h', 'c'], **kwargs)
        self.status.h = np.array([None] * self.params.stencil_length)
        self.status.c = np.array([None] * self.params.stencil_length)

    def compute_work_score(self, work):
        # TODO: docs
        return np.sum([work[key] for key in self.params.work_keys])

    def get_new_step_size(self, controller, S, **kwargs):
        # TODO: docs
        if S.status.iter == S.params.maxiter:

            # update recorded cost
            for i in range(self.params.stencil_length - 1):
                self.status.c[i] = self.status.c[i + 1]
                self.status.h[i] = self.status.h[i + 1]

            self.record_work(S)
            self.status.c[-1] = self.compute_work_score(self.status.work_counters)
            self.status.h[-1] = S.dt

            # compute work derivative here
            if self.status.h[-2] is not None and not np.isclose(self.status.h[-2], self.status.h[-1], atol=1e-8):
                # get finite difference stencil
                mask = self.status.h != None
                _steps, idx = np.unique(
                    -np.cumsum([me - self.status.h[-1] for me in self.status.h if me is not None][::-1]),
                    return_index=True,
                )
                # idx = np.sort(idx)
                # _steps = np.sort(_steps)
                # _steps, idx = np.unique(self.status.h[mask], return_index=True)
                coeff, steps = get_finite_difference_stencil(derivative=1, order=None, type=None, steps=_steps)

                # delta = (np.log(c) - np.log(c_)) / (np.log(h) - np.log(h_))  # from the paper, don't quite understand this

                delta = sum(
                    [
                        coeff[-i] * self.status.c[mask][idx][-i] / self.status.h[mask][idx][-i]
                        for i in range(1, len(coeff) + 1)
                    ]
                )
                # delta = (self.status.c[-1] / self.status.h[-1] - self.status.c[-2] / self.status.h[-2]) / (self.status.h[-1] - self.status.h[-2])  # special case for stencil length 2

                if np.isfinite(delta) and delta > 0:
                    dt_new = S.dt * max([0.1, np.exp(-self.params.gamma * delta)])
                    S.levels[0].status.dt_new = min([dt_new, S.levels[0].status.dt_new])
