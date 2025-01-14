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


@pytest.mark.firedrake
@pytest.mark.parametrize('use_transport_scheme', [True, False])
def test_generic_gusto(use_transport_scheme):
    """
    Use Williamson test Case 5 (flow over a mountain) of Williamson et al, 1992:
    ``A standard test set for numerical approximations to the shallow water
    equations in spherical geometry'', JCP.

    The example here uses the icosahedral sphere mesh and degree 1 spaces.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, setup_equation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    from firedrake import SpatialCoordinate, as_vector, pi, sqrt, min_value, Function
    from gusto import (
        Domain,
        IO,
        OutputParameters,
        DGUpwind,
        ShallowWaterParameters,
        ShallowWaterEquations,
        Sum,
        lonlatr_from_xyz,
        GeneralIcosahedralSphereMesh,
        ZonalComponent,
        MeridionalComponent,
        RelativeVorticity,
        ThetaMethod,
        Timestepper,
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
    dt = 4000
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

    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D", advective_then_flux=True)]
    spatial_methods = None

    if use_transport_scheme:
        eqns = setup_equation(eqns, transport_methods)
        spatial_methods = transport_methods

    problem = GenericGusto(eqns)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u_start = problem.dtype_u(problem.init)
    u0, D0 = u_start.subfunctions[:]

    uexpr = as_vector([-u_max * y / radius, u_max * x / radius, 0.0])
    Dexpr = mean_depth - tpexpr - (radius * Omega * u_max + 0.5 * u_max**2) * (z / radius) ** 2 / g

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    # ------------------------------------------------------------------------ #
    # Prepare gusto reference methods
    # ------------------------------------------------------------------------ #

    stepper_backward = get_gusto_stepper(eqns, ThetaMethod(domain, theta=1.0), spatial_methods)
    stepper_forward = get_gusto_stepper(eqns, ThetaMethod(domain, theta=0.0), spatial_methods)

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
    print(error, test_error)

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
    print(error, test_error)

    assert (
        error < np.finfo(float).eps * 1e2
    ), f'Backward Euler does not match reference implementation! Got relative difference of {error}'


if __name__ == '__main__':
    test_generic_gusto(False)
