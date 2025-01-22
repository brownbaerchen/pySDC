import pytest


def get_gusto_stepper(eqns, method, spatial_methods):
    from gusto import IO, OutputParameters, PrescribedTransport
    import sys

    if '--running-tests' not in sys.argv:
        sys.argv.append('--running-tests')

    output = OutputParameters(dirname='./tmp', dumpfreq=15)
    io = IO(method.domain, output)
    return PrescribedTransport(eqns, method, io, False, transport_method=spatial_methods)


def tracer_setup(tmpdir='./tmp', degree=1, small_dt=False):
    from firedrake import (
        IcosahedralSphereMesh,
        PeriodicIntervalMesh,
        ExtrudedMesh,
        SpatialCoordinate,
        as_vector,
        sqrt,
        exp,
        pi,
    )
    from gusto import OutputParameters, Domain, IO
    from collections import namedtuple

    opts = ('domain', 'tmax', 'io', 'f_init', 'f_end', 'degree', 'uexpr', 'umax', 'radius', 'tol')
    TracerSetup = namedtuple('TracerSetup', opts)
    TracerSetup.__new__.__defaults__ = (None,) * len(opts)

    radius = 1
    mesh = IcosahedralSphereMesh(radius=radius, refinement_level=3, degree=1)
    x = SpatialCoordinate(mesh)

    # Parameters chosen so that dt != 1
    # Gaussian is translated from (lon=pi/2, lat=0) to (lon=0, lat=0)
    # to demonstrate that transport is working correctly
    if small_dt:
        dt = pi / 3.0 * 0.005
    else:
        dt = pi / 3.0 * 0.02

    output = OutputParameters(dirname=str(tmpdir), dumpfreq=15)
    domain = Domain(mesh, dt, family="BDM", degree=degree)
    io = IO(domain, output)

    umax = 1.0
    uexpr = as_vector([-umax * x[1] / radius, umax * x[0] / radius, 0.0])

    tmax = pi / 2
    f_init = exp(-x[2] ** 2 - x[0] ** 2)
    f_end = exp(-x[2] ** 2 - x[1] ** 2)

    tol = 0.05

    return TracerSetup(domain, tmax, io, f_init, f_end, degree, uexpr, umax, radius, tol)


@pytest.fixture
def setup():
    return tracer_setup()


def get_gusto_advection_setup(use_transport_scheme, imex, setup):
    from gusto import ContinuityEquation, AdvectionEquation, split_continuity_form, DGUpwind

    domain = setup.domain
    V = domain.spaces("DG")

    eqn = ContinuityEquation(domain, V, "f")
    eqn = split_continuity_form(eqn)

    transport_methods = [DGUpwind(eqn, 'f')]
    spatial_methods = None

    if use_transport_scheme:
        spatial_methods = transport_methods

    if imex:
        from gusto.core.labels import time_derivative, transport, implicit, explicit

        eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        eqn.label_terms(lambda t: t.has_label(transport), explicit)

    return eqn, domain, spatial_methods, setup


def get_initial_conditions(timestepper, setup):
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)
    return timestepper


class Method(object):
    imex = False

    @staticmethod
    def get_pySDC_method():
        raise NotImplementedError

    @staticmethod
    def get_Gusto_method():
        raise NotImplementedError


class RK4(Method):

    @staticmethod
    def get_pySDC_method():
        from pySDC.implementations.sweeper_classes.Runge_Kutta import RK4

        return RK4

    @staticmethod
    def get_Gusto_method():
        from gusto import RK4

        return RK4


class ImplicitMidpoint(Method):

    @staticmethod
    def get_pySDC_method():
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ImplicitMidpointMethod

        return ImplicitMidpointMethod

    @staticmethod
    def get_Gusto_method():
        from gusto import ImplicitMidpoint

        return ImplicitMidpoint


class BackwardEuler(Method):

    @staticmethod
    def get_pySDC_method():
        from pySDC.implementations.sweeper_classes.Runge_Kutta import BackwardEuler

        return BackwardEuler

    @staticmethod
    def get_Gusto_method():
        from gusto import BackwardEuler

        return BackwardEuler


@pytest.mark.firedrake
@pytest.mark.parametrize('use_transport_scheme', [True, False])
@pytest.mark.parametrize('method', [RK4, ImplicitMidpoint, BackwardEuler])
def test_pySDC_integrator_RK(use_transport_scheme, method, setup):
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.Runge_Kutta import ImplicitMidpointMethod
    from pySDC.implementations.sweeper_classes.Runge_Kutta import RK4 as RK4_pySDC
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
    from gusto import ImplicitMidpoint, RK4
    from firedrake import norm
    import numpy as np

    eqns, domain, spatial_methods, setup = get_gusto_advection_setup(use_transport_scheme, method.imex, setup)

    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
        'ksp_rtol': 1e-12,
        'snes_rtol': 1e-12,
        'ksp_atol': 1e-30,
        'snes_atol': 1e-30,
        'ksp_divtol': 1e30,
        'snes_divtol': 1e30,
        'snes_max_it': 99,
    }

    # ------------------------------------------------------------------------ #
    # Setup pySDC
    # ------------------------------------------------------------------------ #

    level_params = dict()
    level_params['restol'] = -1
    level_params['residual_type'] = 'full_rel'

    step_params = dict()
    step_params['maxiter'] = 1

    sweeper_params = dict()

    problem_params = dict()

    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['mssdc_jac'] = False

    description = dict()
    description['problem_params'] = problem_params
    description['sweeper_class'] = method.get_pySDC_method()
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # ------------------------------------------------------------------------ #
    # Setup time steppers
    # ------------------------------------------------------------------------ #

    stepper_gusto = get_gusto_stepper(
        eqns, method.get_Gusto_method()(domain, solver_parameters=solver_parameters), spatial_methods
    )
    stepper_pySDC = get_gusto_stepper(
        eqns,
        pySDC_integrator(
            eqns,
            description,
            controller_params,
            domain,
            solver_parameters=solver_parameters,
            imex=method.imex,
        ),
        spatial_methods,
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    for stepper in [stepper_gusto, stepper_pySDC]:
        get_initial_conditions(stepper, setup)

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    def run(stepper, n_steps):
        stepper.run(t=0, tmax=n_steps * float(domain.dt))

    for stepper in [stepper_gusto, stepper_pySDC]:
        run(stepper, 5)

    error = norm(stepper_gusto.fields('u') - stepper_pySDC.fields('u')) / norm(stepper_gusto.fields('u'))
    print(error)

    assert (
        error < solver_parameters['snes_rtol'] * 1e3
    ), f'pySDC and Gusto differ in method {method}! Got relative difference of {error}'


# @pytest.mark.firedrake
# @pytest.mark.parametrize('use_transport_scheme', [True, False])
# @pytest.mark.parametrize('IMEX', [True, False])
# def test_pySDC_integrator(use_transport_scheme, IMEX):
#     from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, setup_equation
#     from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
#     from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
#     from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
#     from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
#     from gusto import BackwardEuler, SDC
#     from gusto.core.labels import explicit, implicit, time_derivative, transport
#     from firedrake import norm
#     import numpy as np
#
#     # ------------------------------------------------------------------------ #
#     # Get shallow water setup
#     # ------------------------------------------------------------------------ #
#     dt = 450 if IMEX else 900
#     eqns, domain, spatial_methods, dt, u_start, u0, D0 = get_gusto_SWE_setup(use_transport_scheme, dt=dt)
#     eqns = setup_equation(eqns, spatial_methods=spatial_methods if spatial_methods is not None else [])
#     if IMEX:
#         eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
#         eqns.label_terms(lambda t: t.has_label(transport), explicit)
#         sweeper_class = imex_1st_order
#     else:
#         eqns.label_terms(lambda t: not t.has_label(time_derivative), implicit)
#         sweeper_class = generic_implicit
#
#     # ------------------------------------------------------------------------ #
#     # Setup pySDC
#     # ------------------------------------------------------------------------ #
#     solver_parameters = {
#         'snes_type': 'newtonls',
#         'ksp_type': 'gmres',
#         'pc_type': 'bjacobi',
#         'sub_pc_type': 'ilu',
#         'ksp_rtol': 1e-12,
#         'snes_rtol': 1e-12,
#         'ksp_atol': 1e-30,
#         'snes_atol': 1e-30,
#         'ksp_divtol': 1e30,
#         'snes_divtol': 1e30,
#         'snes_max_it': 999,
#         'ksp_max_it': 999,
#     }
#
#     level_params = dict()
#     level_params['restol'] = -1
#     level_params['residual_type'] = 'full_rel'
#
#     step_params = dict()
#     step_params['maxiter'] = 3
#
#     sweeper_params = dict()
#     sweeper_params['quad_type'] = 'RADAU-RIGHT'
#     sweeper_params['node_type'] = 'LEGENDRE'
#     sweeper_params['num_nodes'] = 2
#     sweeper_params['QI'] = 'IE'
#     sweeper_params['QE'] = 'PIC'
#     sweeper_params['initial_guess'] = 'copy'
#
#     problem_params = dict()
#
#     controller_params = dict()
#     controller_params['logger_level'] = 20
#     controller_params['hook_class'] = []
#     controller_params['mssdc_jac'] = False
#
#     description = dict()
#     description['problem_class'] = GenericGusto
#     description['problem_params'] = problem_params
#     description['sweeper_class'] = sweeper_class
#     description['sweeper_params'] = sweeper_params
#     description['level_params'] = level_params
#     description['step_params'] = step_params
#
#     # ------------------------------------------------------------------------ #
#     # Setup SDC in gusto
#     # ------------------------------------------------------------------------ #
#
#     SDC_params = {
#         'base_scheme': BackwardEuler(domain, solver_parameters=solver_parameters),
#         'M': sweeper_params['num_nodes'],
#         'maxk': step_params['maxiter'],
#         'quad_type': sweeper_params['quad_type'],
#         'node_type': sweeper_params['node_type'],
#         'qdelta_imp': sweeper_params['QI'],
#         'qdelta_exp': sweeper_params['QE'],
#         'formulation': 'Z2N',
#         'initial_guess': 'copy',
#         'nonlinear_solver_parameters': solver_parameters,
#         'linear_solver_parameters': solver_parameters,
#         'final_update': False,
#     }
#
#     # ------------------------------------------------------------------------ #
#     # Setup time steppers
#     # ------------------------------------------------------------------------ #
#
#     stepper_gusto = get_gusto_stepper(eqns, SDC(**SDC_params, domain=domain), spatial_methods)
#     stepper_pySDC = get_gusto_stepper(
#         eqns,
#         pySDC_integrator(
#             eqns,
#             description,
#             controller_params,
#             domain,
#             solver_parameters=solver_parameters,
#             spatial_methods=spatial_methods,
#         ),
#         spatial_methods,
#     )
#
#     # ------------------------------------------------------------------------ #
#     # Run tests
#     # ------------------------------------------------------------------------ #
#
#     assert np.allclose(stepper_gusto.scheme.nodes / dt, stepper_pySDC.scheme.sweeper.coll.nodes)
#     assert np.allclose(stepper_gusto.scheme.Q / dt, stepper_pySDC.scheme.sweeper.coll.Qmat[1:, 1:])
#     assert np.allclose(stepper_gusto.scheme.Qdelta_imp / dt, stepper_pySDC.scheme.sweeper.QI[1:, 1:])
#
#     def run(stepper, n_steps):
#         stepper.fields("u").assign(u0)
#         stepper.fields("D").assign(D0)
#         stepper.run(t=0, tmax=n_steps * dt)
#
#     for stepper in [stepper_gusto, stepper_pySDC]:
#         run(stepper, 10)
#
#     print(
#         norm(stepper_gusto.fields('u') - u0) / norm(stepper_gusto.fields('u')),
#         norm(stepper_pySDC.fields('u') - u0) / norm(stepper_gusto.fields('u')),
#     )
#
#     error = max(
#         [
#             norm(stepper_gusto.fields(comp) - stepper_pySDC.fields(comp)) / norm(stepper_gusto.fields(comp))
#             for comp in ['u', 'D']
#         ]
#     )
#     print(error)
#
#     assert (
#         error < solver_parameters['snes_rtol'] * 1e4
#     ), f'SDC does not match reference implementation! Got relative difference of {error}'
#
#
# @pytest.mark.firedrake
# @pytest.mark.parametrize('IMEX', [True, False])
# @pytest.mark.parametrize('dt', [50, 500])
# def test_pySDC_integrator_with_adaptivity(IMEX, dt):
#     from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, setup_equation
#     from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
#     from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
#     from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
#     from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
#     from pySDC.implementations.convergence_controller_classes.spread_step_sizes import SpreadStepSizesBlockwiseNonMPI
#     from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeRounding
#     from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
#     from pySDC.helpers.stats_helper import get_sorted
#
#     from gusto import BackwardEuler, SDC
#     from gusto.core.labels import explicit, implicit, time_derivative, transport
#     from firedrake import norm, Constant
#     import numpy as np
#
#     # ------------------------------------------------------------------------ #
#     # Get shallow water setup
#     # ------------------------------------------------------------------------ #
#     use_transport_scheme = True
#
#     eqns, domain, spatial_methods, dt, u_start, u0, D0 = get_gusto_SWE_setup(use_transport_scheme, dt=dt)
#     eqns = setup_equation(eqns, spatial_methods=spatial_methods if spatial_methods is not None else [])
#     if IMEX:
#         eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
#         eqns.label_terms(lambda t: t.has_label(transport), explicit)
#         sweeper_class = imex_1st_order
#     else:
#         eqns.label_terms(lambda t: not t.has_label(time_derivative), implicit)
#         sweeper_class = generic_implicit
#
#     # ------------------------------------------------------------------------ #
#     # Setup pySDC
#     # ------------------------------------------------------------------------ #
#     solver_parameters = {
#         'snes_type': 'newtonls',
#         'ksp_type': 'gmres',
#         'pc_type': 'bjacobi',
#         'sub_pc_type': 'ilu',
#         'ksp_rtol': 1e-12,
#         'snes_rtol': 1e-12,
#         'ksp_atol': 1e-30,
#         'snes_atol': 1e-30,
#         'ksp_divtol': 1e30,
#         'snes_divtol': 1e30,
#         'snes_max_it': 999,
#         'ksp_max_it': 999,
#     }
#
#     level_params = dict()
#     level_params['restol'] = -1
#     level_params['residual_type'] = 'full_rel'
#
#     step_params = dict()
#     step_params['maxiter'] = 3
#
#     sweeper_params = dict()
#     sweeper_params['quad_type'] = 'RADAU-RIGHT'
#     sweeper_params['node_type'] = 'LEGENDRE'
#     sweeper_params['num_nodes'] = 2
#     sweeper_params['QI'] = 'IE'
#     sweeper_params['QE'] = 'PIC'
#     sweeper_params['initial_guess'] = 'copy'
#
#     problem_params = dict()
#
#     convergence_controllers = {}
#     convergence_controllers[Adaptivity] = {'e_tol': 1e-6, 'rel_error': True, 'dt_max': 1e4}
#     convergence_controllers[SpreadStepSizesBlockwiseNonMPI] = {'overwrite_to_reach_Tend': False}
#     convergence_controllers[StepSizeRounding] = {}
#
#     controller_params = dict()
#     controller_params['logger_level'] = 20
#     controller_params['hook_class'] = []
#     controller_params['mssdc_jac'] = False
#
#     description = dict()
#     description['problem_class'] = GenericGusto
#     description['problem_params'] = problem_params
#     description['sweeper_class'] = sweeper_class
#     description['sweeper_params'] = sweeper_params
#     description['level_params'] = level_params
#     description['step_params'] = step_params
#     description['convergence_controllers'] = convergence_controllers
#
#     # ------------------------------------------------------------------------ #
#     # Setup SDC in gusto
#     # ------------------------------------------------------------------------ #
#
#     SDC_params = {
#         'base_scheme': BackwardEuler(domain, solver_parameters=solver_parameters),
#         'M': sweeper_params['num_nodes'],
#         'maxk': step_params['maxiter'],
#         'quad_type': sweeper_params['quad_type'],
#         'node_type': sweeper_params['node_type'],
#         'qdelta_imp': sweeper_params['QI'],
#         'qdelta_exp': sweeper_params['QE'],
#         'formulation': 'Z2N',
#         'initial_guess': 'copy',
#         'nonlinear_solver_parameters': solver_parameters,
#         'linear_solver_parameters': solver_parameters,
#         'final_update': False,
#     }
#
#     # ------------------------------------------------------------------------ #
#     # Setup time steppers
#     # ------------------------------------------------------------------------ #
#
#     stepper_gusto = get_gusto_stepper(eqns, SDC(**SDC_params, domain=domain), spatial_methods)
#     stepper_pySDC = get_gusto_stepper(
#         eqns,
#         pySDC_integrator(
#             eqns,
#             description,
#             controller_params,
#             domain,
#             solver_parameters=solver_parameters,
#             spatial_methods=spatial_methods,
#         ),
#         spatial_methods,
#     )
#     stepper_pySDC.scheme.timestepper = stepper_pySDC
#
#     # ------------------------------------------------------------------------ #
#     # Run tests
#     # ------------------------------------------------------------------------ #
#
#     # run with pySDC first
#     stepper_pySDC.fields("u").assign(u0)
#     stepper_pySDC.fields("D").assign(D0)
#     stepper_pySDC.run(t=0, tmax=500)
#
#     # retrieve step sizes
#     stats = stepper_pySDC.scheme.stats
#     dts_pySDC = get_sorted(stats, type='dt', recomputed=False)
#
#     # run with Gusto using same step sizes
#     stepper_gusto.fields("u").assign(u0)
#     stepper_gusto.fields("D").assign(D0)
#     old_dt = float(stepper_gusto.dt)
#
#     for _dt in dts_pySDC:
#         # update step size
#         stepper_gusto.dt = Constant(_dt[1])
#
#         stepper_gusto.scheme.Q *= float(_dt[1] / old_dt)
#         stepper_gusto.scheme.Qdelta_imp *= float(_dt[1] / old_dt)
#         stepper_gusto.scheme.Qdelta_exp *= float(_dt[1] / old_dt)
#         stepper_gusto.scheme.nodes *= float(_dt[1] / old_dt)
#
#         old_dt = _dt[1] * 1.0
#
#         # run
#         stepper_gusto.run(t=_dt[0], tmax=_dt[0] + _dt[1])
#
#     assert np.isclose(float(stepper_pySDC.t), float(stepper_gusto.t))
#
#     print(
#         norm(stepper_gusto.fields('u') - u0) / norm(stepper_gusto.fields('u')),
#         norm(stepper_pySDC.fields('u') - u0) / norm(stepper_gusto.fields('u')),
#     )
#     print(dts_pySDC)
#
#     error = max(
#         [
#             norm(stepper_gusto.fields(comp) - stepper_pySDC.fields(comp)) / norm(stepper_gusto.fields(comp))
#             for comp in ['u', 'D']
#         ]
#     )
#
#     # compute round-off error: Warning: It's large!
#     roundoff_Q = max([np.max(np.abs(stepper_gusto.scheme.Q - stepper_pySDC.scheme.sweeper.coll.Qmat[1:, 1:] * _dt[1])), np.finfo(float).eps*1e2])
#     roundoff = roundoff_Q * max(norm(stepper_gusto.fields(comp)) for comp in ['u', 'D'])
#
#     print(error, roundoff)
#
#     assert (
#         error < roundoff * 1e1
#     ), f'SDC does not match reference implementation with adaptive step size selection! Got relative difference of {error}, while round-off error is roughly {roundoff:.2e}'


if __name__ == '__main__':
    setup = tracer_setup()
    test_pySDC_integrator_RK(False, RK4, setup)
    exit()
    # test_generic_gusto(True)
    # test_pySDC_integrator_RK(True, 'BackwardEuler')
    # test_pySDC_integrator_RK(False, 'ImplicitMidpoint')
    # test_pySDC_integrator(False, True)
    # test_pySDC_integrator_with_adaptivity(False, 500)
