import pytest


def get_gusto_stepper(eqns, method, spatial_methods):
    from gusto import Timestepper, IO, OutputParameters, Sum, MeridionalComponent, RelativeVorticity, ZonalComponent
    import numpy as np

    # TODO: Can I get rid of the output here?

    output = OutputParameters(
        dirname=f'./tmp{np.random.randint(1e9)}',
        dumplist_latlon=['D'],
        dumpfreq=1000000000,
        dump_vtus=True,
        dump_nc=False,
        dumplist=['D', 'topography'],
    )
    diagnostic_fields = [Sum('D', 'topography'), RelativeVorticity(), MeridionalComponent('u'), ZonalComponent('u')]
    io = IO(method.domain, output, diagnostic_fields=diagnostic_fields)
    return Timestepper(eqns, method, io, spatial_methods=spatial_methods)


def get_gusto_SWE_setup(use_transport_scheme, dt=4000):
    """
    Use Williamson test Case 5 (flow over a mountain) of Williamson et al, 1992:
    ``A standard test set for numerical approximations to the shallow water
    equations in spherical geometry'', JCP.

    The example here uses the icosahedral sphere mesh and degree 1 spaces.
    """
    from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, setup_equation

    from firedrake import SpatialCoordinate, as_vector, pi, sqrt, min_value
    from gusto import (
        Domain,
        DGUpwind,
        ShallowWaterParameters,
        ShallowWaterEquations,
        lonlatr_from_xyz,
        GeneralIcosahedralSphereMesh,
        ThetaMethod,
    )

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.0  # planetary radius (m)
    mean_depth = 5960  # reference depth (m)
    g = 9.80616  # acceleration due to gravity (m/s^2)
    u_max = 20.0  # max amplitude of the zonal wind (m/s)
    mountain_height = 2000.0  # height of mountain (m)
    R0 = pi / 9.0  # radius of mountain (rad)
    lamda_c = -pi / 2.0  # longitudinal centre of mountain (rad)
    phi_c = pi / 6.0  # latitudinal centre of mountain (rad)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    ncells_per_edge = 16
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, 'BDM', element_order)
    x, y, z = SpatialCoordinate(mesh)
    lamda, phi, _ = lonlatr_from_xyz(x, y, z)

    # Equation: coriolis
    parameters = ShallowWaterParameters(H=mean_depth, g=g)
    Omega = parameters.Omega
    fexpr = 2 * Omega * z / radius

    # Equation: topography
    rsq = min_value(R0**2, (lamda - lamda_c) ** 2 + (phi - phi_c) ** 2)
    r = sqrt(rsq)
    tpexpr = mountain_height * (1 - r / R0)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, topog_expr=tpexpr)
    eqns_with_transport = ShallowWaterEquations(domain, parameters, fexpr=fexpr, topog_expr=tpexpr)

    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D", advective_then_flux=True)]
    spatial_methods = None

    if use_transport_scheme:
        eqns_with_transport = setup_equation(eqns_with_transport, transport_methods)
        spatial_methods = transport_methods

    problem = GenericGusto(eqns_with_transport)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u_start = problem.dtype_u(problem.init)
    u0, D0 = u_start.subfunctions[:]

    uexpr = as_vector([-u_max * y / radius, u_max * x / radius, 0.0])
    Dexpr = mean_depth - tpexpr - (radius * Omega * u_max + 0.5 * u_max**2) * (z / radius) ** 2 / g

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    return eqns, eqns_with_transport, domain, spatial_methods, dt, u_start, u0, D0


@pytest.mark.firedrake
@pytest.mark.parametrize('use_transport_scheme', [True, False])
def test_generic_gusto(use_transport_scheme):
    from pySDC.implementations.problem_classes.GenericGusto import GenericGusto
    from gusto import ThetaMethod
    import numpy as np

    # ------------------------------------------------------------------------ #
    # Get shallow water setup
    # ------------------------------------------------------------------------ #
    eqns, eqns_with_transport, domain, spatial_methods, dt, u_start, u0, D0 = get_gusto_SWE_setup(use_transport_scheme)

    # ------------------------------------------------------------------------ #
    # Prepare different methods
    # ------------------------------------------------------------------------ #

    problem = GenericGusto(eqns_with_transport)
    stepper_backward = get_gusto_stepper(eqns, ThetaMethod(domain, theta=1.0), spatial_methods)
    stepper_forward = get_gusto_stepper(eqns, ThetaMethod(domain, theta=0.0), spatial_methods)
    print(' ----- pySDC -----')
    print(problem.residual.form.__str__())
    print(' ----- Gusto -----')
    print(stepper_backward.scheme.residual.form.__str__())
    print(problem.residual.form.__str__() == stepper_backward.scheme.residual.form.__str__())
    exit()

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    un = problem.solve_system(u_start, dt, u_start)
    fn = problem.eval_f(un)

    u02 = un - dt * fn

    error = abs(u_start - u02) / abs(u_start)
    test_error = abs(u_start - un) / abs(u_start)

    assert error < 5e-2 * test_error
    print(error)

    # test forward Euler step
    stepper_forward.fields("u").assign(u0)
    stepper_forward.fields("D").assign(D0)
    stepper_forward.run(t=0, tmax=dt)
    un_ref = problem.dtype_u(problem.init)
    u, D = un_ref.subfunctions[:]
    u.assign(stepper_forward.fields('u'))
    D.assign(stepper_forward.fields('D'))
    un_forward = u_start + dt * problem.eval_f(u_start)
    error = abs(un_forward - un_ref) / abs(un_ref)
    print(error)

    assert (
        error < np.finfo(float).eps * 1e2
    ), f'Forward Euler does not match reference implementation! Got relative difference of {error}'

    # test backward Euler step
    stepper_backward.fields("u").assign(u0)
    stepper_backward.fields("D").assign(D0)
    stepper_backward.run(t=0, tmax=dt)
    un_ref = problem.dtype_u(problem.init)
    u, D = un_ref.subfunctions[:]
    u.assign(stepper_backward.fields('u'))
    D.assign(stepper_backward.fields('D'))
    error = abs(un - un_ref) / abs(un_ref)
    print(error)

    assert (
        error < np.finfo(float).eps * 1e2
    ), f'Backward Euler does not match reference implementation! Got relative difference of {error}'


@pytest.mark.firedrake
@pytest.mark.parametrize('use_transport_scheme', [True, False])
@pytest.mark.parametrize('method', ['RK4', 'ImplicitMidpoint', 'BackwardEuler'])
def test_pySDC_integrator_RK(use_transport_scheme, method):
    from pySDC.implementations.problem_classes.GenericGusto import GenericGusto
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.sweeper_classes.Runge_Kutta import ImplicitMidpointMethod
    from pySDC.implementations.sweeper_classes.Runge_Kutta import RK4 as RK4_pySDC
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
    from gusto import ImplicitMidpoint, RK4
    from firedrake import norm
    import numpy as np

    if method == 'RK4':
        from pySDC.implementations.sweeper_classes.Runge_Kutta import RK4 as pySDC_method
        from gusto import RK4 as gusto_method

        dt = 450
    elif method == 'ImplicitMidpoint':
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ImplicitMidpointMethod as pySDC_method
        from gusto import ImplicitMidpoint as gusto_method

        dt = 100
    elif method == 'BackwardEuler':
        from pySDC.implementations.sweeper_classes.Runge_Kutta import BackwardEuler as pySDC_method
        from gusto import BackwardEuler as gusto_method

        dt = 900
    else:
        raise NotImplementedError

    # ------------------------------------------------------------------------ #
    # Get shallow water setup
    # ------------------------------------------------------------------------ #
    eqns, eqns_with_transport, domain, spatial_methods, dt, u_start, u0, D0 = get_gusto_SWE_setup(use_transport_scheme, dt=dt)

    # ------------------------------------------------------------------------ #
    # Setup pySDC
    # ------------------------------------------------------------------------ #
    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'ksp_rtol': 1e-12,
        'snes_rtol': 1e-12,
        'ksp_atol': 1e-30,
        'snes_atol': 1e-30,
        'ksp_divtol': 1e30,
        'snes_divtol': 1e30,
    }

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
    description['sweeper_class'] = pySDC_method
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # ------------------------------------------------------------------------ #
    # Setup time steppers
    # ------------------------------------------------------------------------ #

    stepper_gusto = get_gusto_stepper(eqns, gusto_method(domain, solver_parameters=solver_parameters), spatial_methods)
    stepper_pySDC = get_gusto_stepper(
        eqns,
        pySDC_integrator(eqns, description, controller_params, domain, solver_parameters=solver_parameters),
        spatial_methods,
    )

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    def run(stepper, n_steps):
        stepper.fields("u").assign(u0)
        stepper.fields("D").assign(D0)
        stepper.run(t=0, tmax=n_steps * dt)

    for stepper in [stepper_gusto, stepper_pySDC]:
        run(stepper, 3)

    print(
        norm(stepper_gusto.fields('u') - u0) / norm(stepper_gusto.fields('u')),
        norm(stepper_pySDC.fields('u') - u0) / norm(stepper_gusto.fields('u')),
    )

    error = max(
        [
            norm(stepper_gusto.fields(comp) - stepper_pySDC.fields(comp)) / norm(stepper_gusto.fields(comp))
            for comp in ['u', 'D']
        ]
    )
    print(error)

    assert (
        error < np.finfo(float).eps * 1e2
    ), f'pySDC and Gusto differ in method {method}! Got relative difference of {error}'


@pytest.mark.firedrake
@pytest.mark.parametrize('use_transport_scheme', [True, False])
def test_pySDC_integrator(use_transport_scheme):
    from pySDC.implementations.problem_classes.GenericGusto import GenericGusto
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
    from gusto import BackwardEuler, SDC
    from gusto.core.labels import explicit, implicit, time_derivative
    from firedrake import norm
    import numpy as np

    # ------------------------------------------------------------------------ #
    # Get shallow water setup
    # ------------------------------------------------------------------------ #
    eqns, eqns_with_transport, domain, spatial_methods, dt, u_start, u0, D0 = get_gusto_SWE_setup(use_transport_scheme, dt=900)
    eqns.label_terms(lambda t: not t.has_label(time_derivative), implicit)

    # ------------------------------------------------------------------------ #
    # Setup pySDC
    # ------------------------------------------------------------------------ #
    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'ksp_rtol': 1e-14,
        'snes_rtol': 1e-14,
        'ksp_atol': 1e-30,
        'snes_atol': 1e-30,
        'ksp_divtol': 1e30,
        'snes_divtol': 1e30,
        'snes_max_it': 999,
    }

    level_params = dict()
    level_params['restol'] = -1
    level_params['residual_type'] = 'full_rel'

    step_params = dict()
    step_params['maxiter'] = 3

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['num_nodes'] = 2
    sweeper_params['QI'] = 'IE'
    sweeper_params['QE'] = 'PIC'
    sweeper_params['initial_guess'] = 'copy'

    problem_params = dict()

    from pySDC.implementations.hooks.log_solution import LogSolution

    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = [LogSolution]
    controller_params['mssdc_jac'] = False

    description = dict()
    description['problem_class'] = GenericGusto
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # ------------------------------------------------------------------------ #
    # Setup SDC in gusto
    # ------------------------------------------------------------------------ #

    SDC_params = {
        'base_scheme': BackwardEuler(domain, solver_parameters=solver_parameters),
        'M': sweeper_params['num_nodes'],
        'maxk': step_params['maxiter'],
        'quad_type': sweeper_params['quad_type'],
        'node_type': sweeper_params['node_type'],
        'qdelta_imp': sweeper_params['QI'],
        'qdelta_exp': sweeper_params['QE'],
        'formulation': 'Z2N',
        'initial_guess': 'copy',
        'nonlinear_solver_parameters': solver_parameters,
        'linear_solver_parameters': solver_parameters,
        'final_update': False, 
    }

    # ------------------------------------------------------------------------ #
    # Setup time steppers
    # ------------------------------------------------------------------------ #

    stepper_gusto = get_gusto_stepper(eqns, SDC(**SDC_params, domain=domain), spatial_methods)
    stepper_pySDC = get_gusto_stepper(
        eqns,
        pySDC_integrator(eqns, description, controller_params, domain, solver_parameters=solver_parameters),
        spatial_methods,
    )

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    assert np.allclose(stepper_gusto.scheme.nodes / dt, stepper_pySDC.scheme.sweeper.coll.nodes)
    assert np.allclose(stepper_gusto.scheme.Q / dt, stepper_pySDC.scheme.sweeper.coll.Qmat[1:, 1:])
    assert np.allclose(stepper_gusto.scheme.Qdelta_imp / dt, stepper_pySDC.scheme.sweeper.QI[1:, 1:])

    def run(stepper, n_steps):
        stepper.fields("u").assign(u0)
        stepper.fields("D").assign(D0)
        stepper.run(t=0, tmax=n_steps * dt)

    for stepper in [stepper_gusto, stepper_pySDC]:
        run(stepper, 3)

    print(
        norm(stepper_gusto.fields('u') - u0) / norm(stepper_gusto.fields('u')),
        norm(stepper_pySDC.fields('u') - u0) / norm(stepper_gusto.fields('u')),
    )

    error = max(
        [
            norm(stepper_gusto.fields(comp) - stepper_pySDC.fields(comp)) / norm(stepper_gusto.fields(comp))
            for comp in ['u', 'D']
        ]
    )
    print(error)

    assert (
        error < np.finfo(float).eps * 1e2
    ), f'SDC does not match reference implementation! Got relative difference of {error}'


if __name__ == '__main__':
    test_generic_gusto(True)
    # test_pySDC_integrator_RK(True, 'RK4')
    # test_pySDC_integrator_RK(False, 'ImplicitMidpoint')
    # test_pySDC_integrator(False)
