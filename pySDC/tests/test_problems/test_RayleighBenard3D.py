import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('direction', ['x', 'y', 'z', 'mixed'])
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [8])
@pytest.mark.parametrize('spectral_space', [True, False])
def test_eval_f(nx, nz, direction, spectral_space):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D

    P = RayleighBenard3D(nx=nx, ny=nx, nz=nz, Rayleigh=1, spectral_space=spectral_space)
    iu, iv, iw, ip, iT = P.index(['u', 'v', 'w', 'p', 'T'])
    X, Y, Z = P.X, P.Y, P.Z
    cos, sin = np.cos, np.sin

    kappa = P.kappa
    nu = P.nu

    if direction == 'x':
        y = sin(X * np.pi)
        y_x = cos(X * np.pi) * np.pi
        y_xx = -sin(X * np.pi) * np.pi**2
        y_y = 0
        y_yy = 0
        y_z = 0
        y_zz = 0
    elif direction == 'y':
        y = sin(Y * np.pi)
        y_y = cos(Y * np.pi) * np.pi
        y_yy = -sin(Y * np.pi) * np.pi**2
        y_x = 0
        y_xx = 0
        y_z = 0
        y_zz = 0
    elif direction == 'z':
        y = Z**2
        y_x = 0
        y_xx = 0
        y_y = 0
        y_yy = 0
        y_z = 2 * Z
        y_zz = 2.0
    elif direction == 'mixed':
        y = sin(X * np.pi) * sin(Y * np.pi) * Z**2
        y_x = cos(X * np.pi) * np.pi * sin(Y * np.pi) * Z**2
        y_xx = -sin(X * np.pi) * np.pi**2 * sin(Y * np.pi) * Z**2
        y_y = cos(Y * np.pi) * np.pi * sin(X * np.pi) * Z**2
        y_yy = -sin(Y * np.pi) * np.pi**2 * sin(X * np.pi) * Z**2
        y_z = sin(X * np.pi) * sin(Y * np.pi) * 2 * Z
        y_zz = sin(X * np.pi) * sin(Y * np.pi) * 2
    else:
        raise NotImplementedError

    assert np.allclose(P.eval_f(P.u_init), 0), 'Non-zero time derivative in static 0 configuration'

    u = P.u_init
    for i in [iu, iv, iw, iT, ip]:
        u[i][:] = y

    if spectral_space:
        u = P.transform(u)

    f = P.eval_f(u)

    f_expect = P.f_init
    for i in [iT, iu, iv, iw]:
        f_expect.expl[i] = -y * (y_x + y_y + y_z)
    f_expect.impl[iT] = kappa * (y_xx + +y_yy + y_zz)
    f_expect.impl[iu] = -y_x + nu * (y_xx + y_yy + y_zz)
    f_expect.impl[iv] = -y_y + nu * (y_xx + y_yy + y_zz)
    f_expect.impl[iw] = -y_z + nu * (y_xx + +y_yy + y_zz) + y
    f_expect.impl[ip] = -(y_x + y_y + y_z)

    if spectral_space:
        f.impl = P.itransform(f.impl).real
        f.expl = P.itransform(f.expl).real

    for comp in ['u', 'v', 'T', 'p']:
        i = P.spectral.index(comp)
        assert np.allclose(f.impl[i], f_expect.impl[i]), f'Unexpected implicit function evaluation in component {comp}'
        assert np.allclose(f.expl[i], f_expect.expl[i]), f'Unexpected explicit function evaluation in component {comp}'


#
# @pytest.mark.mpi4py
# @pytest.mark.parametrize('nx', [16])
# @pytest.mark.parametrize('nz', [4])
# @pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
# def test_vorticity(nx, nz, direction):
#     import numpy as np
#     from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D
#
#     assert nz > 3
#     assert nx > 8
#
#     P = RayleighBenard3D(nx=nx, nz=nz, spectral_space=False)
#     iu, iv = P.index(['u', 'v'])
#
#     u = P.u_init
#
#     if direction == 'x':
#         u[iv] = np.sin(P.X * np.pi)
#         u[iu] = np.cos(P.X * np.pi)
#         expect = np.cos(P.X * np.pi) * np.pi
#     elif direction == 'z':
#         u[iv] = P.Z**2
#         u[iu] = P.Z**3
#         expect = 3 * P.Z**2
#     elif direction == 'mixed':
#         u[iv] = np.sin(P.X * np.pi) * P.Z**2
#         u[iu] = np.cos(P.X * np.pi) * P.Z**3
#         expect = np.cos(P.X * np.pi) * np.pi * P.Z**2 + np.cos(P.X * np.pi) * 3 * P.Z**2
#     else:
#         raise NotImplementedError
#
#     assert np.allclose(P.compute_vorticity(u), expect)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [1, 8])
@pytest.mark.parametrize('component', ['u', 'v', 'T'])
def test_Poisson_problems(nx, component):
    """
    When forgetting about convection and the time-dependent part, you get Poisson problems in u and T that are easy to solve. We check that we get the exact solution in a simple test here.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'w_top': 0,
        'w_bottom': 0,
        'T_top': 0,
        'T_bottom': 0,
    }
    P = RayleighBenard3D(
        nx=nx, ny=nx, nz=6, BCs=BCs, Rayleigh=(max([abs(BCs['T_top'] - BCs['T_bottom']), np.finfo(float).eps]) * 2**3)
    )
    rhs = P.u_init

    idx = P.index(f'{component}')

    A = P.put_BCs_in_matrix(-P.L)
    rhs[idx][0, 0, 2] = 6
    rhs[idx][0, 0, 0] = 6
    u = P.sparse_lib.linalg.spsolve(A, P.M @ rhs.flatten()).reshape(rhs.shape).real

    u_exact = P.u_init
    u_exact[idx][0, 0, 4] = 1 / 8
    u_exact[idx][0, 0, 2] = 1 / 2
    u_exact[idx][0, 0, 0] = -5 / 8

    if component == 'T':
        ip = P.index('p')
        u_exact[ip][0, 0, 5] = 1 / (16 * 5)
        u_exact[ip][0, 0, 3] = 5 / (16 * 5)
        u_exact[ip][0, 0, 1] = -70 / (16 * 5)

    assert np.allclose(u_exact, u)


@pytest.mark.mpi4py
def test_Poisson_problem_w():
    """
    Here we don't really solve a Poisson problem. w can only be constant due to the incompressibility, then we have a Possion problem in T with a linear solution and p absorbs all the rest. This is therefore mainly a test for the pressure computation. We don't test that the boundary condition is enforced because the constant pressure offset is entirely irrelevant to anything.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'w_top': 0,
        'w_bottom': 0,
        'T_top': 0,
        'T_bottom': 2,
    }
    P = RayleighBenard3D(nx=2, ny=2, nz=2**3, BCs=BCs, Rayleigh=1.0)
    iw = P.index('w')

    rhs_real = P.u_init
    rhs_real[iw] = 32 * 6 * P.Z**5

    rhs = P.transform(rhs_real)
    rhs = (P.M @ rhs.flatten()).reshape(rhs.shape)
    rhs_real = P.itransform(rhs)

    rhs_real = P.put_BCs_in_rhs(rhs_real)
    rhs = P.transform(rhs_real)

    A = P.put_BCs_in_matrix(-P.L)
    u = P.sparse_lib.linalg.spsolve(A, rhs.flatten()).reshape(rhs.shape).real

    u_exact_real = P.u_init
    iT = P.index('T')
    u_exact_real[iT] = 1 - P.Z

    ip = P.index('p')
    u_exact_real[ip] = P.Z - 1 / 2 * P.Z**2 - 32 * P.Z**6

    u_exact = P.transform(u_exact_real)
    u_exact[ip, 0, 0] = u[ip, 0, 0]  # nobody cares about the constant offset
    assert np.allclose(u_exact, u)


# @pytest.mark.mpi4py
# def test_Nyquist_mode_elimination():
#     from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D
#     import numpy as np
#
#     P = RayleighBenard3D(nx=32, nz=8)
#     u0 = P.u_exact(noise_level=1e-3)
#
#     u = P.solve_system(u0, dt=1e-3)
#
#     Nyquist_mode_index = P.axes[0].get_Nyquist_mode_index()
#     assert np.allclose(u[:, Nyquist_mode_index, :], 0)
#
#
# @pytest.mark.mpi4py
# def test_apply_BCs():
#     from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D
#     import numpy as np
#
#     BCs = {
#         'u_top': np.random.rand(),
#         'u_bottom': np.random.rand(),
#         'v_top': np.random.rand(),
#         'v_bottom': np.random.rand(),
#         'T_top': np.random.rand(),
#         'T_bottom': np.random.rand(),
#     }
#     P = RayleighBenard3D(nx=5, nz=2**2, BCs=BCs)
#
#     u_in = P.u_init
#     u_in[...] = np.random.rand(*u_in.shape)
#     u_in_hat = P.transform(u_in)
#
#     u_hat = P.apply_BCs(u_in_hat)
#     u = P.itransform(u_hat)
#
#     P.check_BCs(u)


if __name__ == '__main__':
    test_eval_f(2**0, 2**2, 'z', True)
    # test_Poisson_problems(3, 'u')
    # test_Poisson_problem_w()
    # test_Nusselt_numbers(1)
    # test_buoyancy_computation()
    # test_viscous_dissipation()
    # test_Nyquist_mode_elimination()
