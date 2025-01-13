import pytest


@pytest.mark.firedrake
def test_generic_gusto():
    """
    Use Williamson test Case 5 (flow over a mountain) of Williamson et al, 1992:
    ``A standard test set for numerical approximations to the shallow water
    equations in spherical geometry'', JCP.

    The example here uses the icosahedral sphere mesh and degree 1 spaces.
    """
    import firedrake as fd
    from pySDC.implementations.problem_classes.GenericGusto import GenericGusto, setup_equation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    from firedrake import SpatialCoordinate, as_vector, pi, sqrt, min_value, Function
    from gusto import (
        Domain,
        IO,
        OutputParameters,
        SemiImplicitQuasiNewton,
        SSPRK3,
        DGUpwind,
        ShallowWaterParameters,
        ShallowWaterEquations,
        Sum,
        lonlatr_from_xyz,
        GeneralIcosahedralSphereMesh,
        ZonalComponent,
        MeridionalComponent,
        RelativeVorticity,
        RungeKuttaFormulation,
        SubcyclingOptions,
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

    # Transport schemes
    # subcycling_options = SubcyclingOptions(subcycle_by_courant=0.25)
    # transported_fields = [
    #     SSPRK3(domain, "u", subcycling_options=subcycling_options),
    #     SSPRK3(
    #         domain, "D", subcycling_options=subcycling_options,
    #         rk_formulation=RungeKuttaFormulation.linear
    #     )
    # ]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D", advective_then_flux=True)]

    eqns_with_transport = setup_equation(eqns, transport_methods)

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

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    un = problem.solve_system(u_start, dt, u_start)
    fn = problem.eval_f(un)

    u02 = un - dt * fn

    error = abs(u_start - u02) / abs(u_start)
    test_error = abs(u_start - un) / abs(u_start)

    assert error < 5e-2 * test_error


if __name__ == '__main__':
    test_generic_gusto()
