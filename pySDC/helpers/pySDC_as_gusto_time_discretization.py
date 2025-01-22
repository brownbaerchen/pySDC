import firedrake as fd

from gusto.timestepping import BaseTimestepper
from gusto.time_discretisation.time_discretisation import TimeDiscretisation, wrapper_apply
from gusto.core.labels import implicit, explicit

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, GenericGustoImex, setup_equation
from pySDC.core.hooks import Hooks
from pySDC.helpers.stats_helper import get_sorted


class LogTime(Hooks):
    def post_step(self, step, level_number):
        L = step.levels[level_number]
        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=-1,
            iter=-1,
            sweep=-1,
            type='_time',
            value=L.time + L.dt,
        )


class pySDC_integrator(TimeDiscretisation):
    def __init__(
        self,
        equation,
        description,
        controller_params,
        domain,
        field_name=None,
        subcycling_options=None,
        solver_parameters=None,
        limiter=None,
        options=None,
        augmentation=None,
        t0=0,
    ):

        self._residual = None

        super().__init__(
            domain=domain,
            field_name=field_name,
            subcycling_options=subcycling_options,
            solver_parameters=solver_parameters,
            limiter=limiter,
            options=options,
            augmentation=augmentation,
        )

        self.description = description
        self.controller_params = controller_params
        self.timestepper = None
        self.dt_next = None

    def setup(self, equation, apply_bcs=True, *active_labels):
        super().setup(equation, apply_bcs, *active_labels)

        # Check if any terms are explicit
        IMEX = any(t.has_label(explicit) for t in equation.residual)
        if IMEX:
            self.description['problem_class'] = GenericGustoImex
        else:
            self.description['problem_class'] = GenericGusto

        self.description['problem_params'] = {
            'equation': equation,
            'solver_parameters': self.solver_parameters,
            'residual': self._residual,
        }
        self.description['level_params']['dt'] = float(self.domain.dt)

        # this is required for step size adaptivity
        hook_class = self.controller_params.get('hook_class', [])
        if not type(hook_class) == list:
            hook_class = [hook_class]
        hook_class.append(LogTime)
        self.controller_params['hook_class'] = hook_class

        # prepare controller and variables
        self.controller = controller_nonMPI(1, description=self.description, controller_params=self.controller_params)
        self.prob = self.level.prob
        self.sweeper = self.level.sweep
        self.x0_pySDC = self.prob.dtype_u(self.prob.init)
        self.t = 0
        self.stats = {}

    @property
    def residual(self):
        if hasattr(self, 'prob'):
            return self.prob.residual
        else:
            return self._residual

    @residual.setter
    def residual(self, value):
        if hasattr(self, 'prob'):
            self.prob.residual = value
        else:
            self._residual = value

    @property
    def level(self):
        """Get the finest pySDC level"""
        return self.controller.MS[0].levels[0]

    @wrapper_apply
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        self.x0_pySDC.functionspace.assign(x_in)
        assert self.level.params.dt == float(self.dt), 'Step sizes have diverged between pySDC and Gusto'

        if self.dt_next is not None:
            assert (
                self.timestepper is not None
            ), 'You need to set self.timestepper to the timestepper in order to facilitate adaptive step size selection here!'
            self.timestepper.dt = fd.Constant(self.dt_next)
            self.t = self.timestepper.t

        uend, _stats = self.controller.run(u0=self.x0_pySDC, t0=float(self.t), Tend=float(self.t + self.dt))

        # update time variables
        if self.level.params.dt != float(self.dt):
            self.dt_next = self.level.params.dt

        self.t = get_sorted(_stats, type='_time', recomputed=False)[-1][1]

        if self.timestepper is not None:
            self.timestepper.t = fd.Constant(self.t - self.dt)

        self.dt = self.level.params.dt

        self.stats = {**self.stats, **_stats}
        x_out.assign(uend.functionspace)
