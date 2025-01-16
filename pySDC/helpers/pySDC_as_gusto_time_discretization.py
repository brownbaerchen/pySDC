from gusto.timestepping import BaseTimestepper
from gusto.time_discretisation.time_discretisation import TimeDiscretisation, wrapper_apply

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, setup_equation


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
        spatial_methods=None,
        t0=0,
    ):
        if spatial_methods is not None:
            equation = setup_equation(equation, spatial_methods=spatial_methods)

        description['problem_class'] = GenericGusto
        description['solver_parameters'] = solver_parameters
        description['problem_params'] = {'equation': equation, 'solver_parameters': solver_parameters}
        description['level_params']['dt'] = float(domain.dt)

        self.controller = controller_nonMPI(1, description=description, controller_params=controller_params)
        self.P = self.controller.MS[0].levels[0].prob
        self.sweeper = self.controller.MS[0].levels[0].sweep
        self.x0_pySDC = self.P.dtype_u(self.P.init)
        self.t = 0
        self.stats = {}

        if not solver_parameters:
            # default solver parameters
            solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
        super().__init__(
            domain=domain,
            field_name=field_name,
            subcycling_options=subcycling_options,
            solver_parameters=solver_parameters,
            limiter=limiter,
            options=options,
            augmentation=augmentation,
        )

    @wrapper_apply
    def apply(self, x_out, x_in):
        """
        Apply the time discretisation to advance one whole time step.

        Args:
            x_out (:class:`Function`): the output field to be computed.
            x_in (:class:`Function`): the input field.
        """
        self.x0_pySDC.functionspace.assign(x_in)
        assert self.controller.MS[0].levels[0].params.dt == float(
            self.dt
        ), 'Step sizes have diverged between pySDC and Gusto'
        uend, _stats = self.controller.run(u0=self.x0_pySDC, t0=self.t, Tend=self.t + float(self.dt))
        self.t += float(self.dt)
        self.stats = {**self.stats, **_stats}
        x_out.assign(uend.functionspace)
