import pytest

PARAMS = {
    'nx': 2**7,
    'nz': 2**7,
}


def IMEX_Euler(_u, dt):
    f = P.eval_f(_u)
    un = P.solve_system(_u + dt * f.expl, dt)
    return un


def implicit_Euler(_u, dt):
    un = P.solve_system(_u, dt)
    return un


def compute_errors(u1, u2, msg, thresh=1e-10, components=None, raise_errors=True, P=None):
    import numpy as np

    components = P.components if components is None else components
    msgs = ''
    for comp in components:
        i = P.index(comp)
        error = abs(u1[i] - u2[i])
        if error > thresh:
            msgs = f'{msgs} {error=:2e} in {comp}'
        if not (np.allclose(u2[i].imag, 0) and np.allclose(u1[i].imag, 0)):
            msgs = f'{msgs} non-zero imaginary part in {comp}'
    if raise_errors:
        assert msgs == '', f'Errors too large when solving {msg}: {msgs}'
    elif msgs != '':
        print(f'Errors too large when solving {msg}: {msgs}')

    violations = P.compute_constraint_violation(u2)
    for key in [key for key in violations.keys() if key in components]:
        if raise_errors:
            assert np.allclose(
                violations[key], 0
            ), f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!'
        elif abs(violations[key]) > 1e-11:
            print(f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!')

    BC_violations = P.compute_BC_violation(u2)
    for key in BC_violations.keys():
        if raise_errors:
            assert np.allclose(
                BC_violations[key], 0
            ), f'Violation of boundary conditions in {key}: {abs(BC_violations[key]):.2e} after solving {msg}!'
        elif not np.allclose(BC_violations[key], 0):
            print(f'Violation of boundary conditions in {key}: {abs(BC_violations[key]):.2e} after solving {msg}!')


@pytest.mark.mpi4py
@pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
def test_derivatives(nx, nz, direction, cheby_mode):
    '''
    Test the derivatives in x- and z-directions.
    '''
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode)

    u = P.u_init
    for i in range(u.shape[0]):
        if direction == 'z':
            u[i] = P.Z ** (nz - 1)
            expect_x = np.zeros_like(P.X)
            expect_z = (nz - 1) * P.Z ** (nz - 2)
        elif direction == 'x':
            u[i] = np.sin(P.X)
            expect_z = np.zeros_like(P.X)
            expect_x = np.cos(P.X)
        elif direction == 'mixed':
            u[i] = np.sin(P.X) * P.Z**2 + P.Z ** (nz - 1)
            expect_z = (nz - 1) * P.Z ** (nz - 2) + 2 * P.Z * np.sin(P.X)
            expect_x = np.cos(P.X) * P.Z ** (2)
        else:
            raise NotImplementedError

    derivatives = P.compute_z_derivatives(u)

    for comp in ['vz', 'Tz']:
        i = P.spectral.index(comp)
        assert np.allclose(derivatives[i], expect_z), f'Got unexpected z-derivative in component {comp}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
@pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [8])
@pytest.mark.parametrize('compute_violations', [True, False])
def test_eval_f(nx, nz, cheby_mode, direction, compute_violations):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode)
    iu, iv, ip, iT, ivz, iuz, iTz = P.index(['u', 'v', 'p', 'T', 'vz', 'uz', 'Tz'])
    X, Z = P.X, P.Z
    cos, sin = np.cos, np.sin

    kappa = (P.Rayleigh * P.Prandl) ** (-1 / 2)
    nu = (P.Rayleigh / P.Prandl) ** (-1 / 2)

    if direction == 'x':
        y = sin(X * np.pi)
        y_x = cos(X * np.pi) * np.pi
        y_xx = -sin(X * np.pi) * np.pi**2
        y_z = 0
        y_zz = 0
    elif direction == 'z':
        y = Z**2
        y_x = 0
        y_xx = 0
        y_z = 2 * Z
        y_zz = 2.0
    elif direction == 'mixed':
        y = sin(X * np.pi) * Z**2
        y_x = cos(X * np.pi) * np.pi * Z**2
        y_xx = -sin(X * np.pi) * np.pi**2 * Z**2
        y_z = sin(X * np.pi) * 2 * Z
        y_zz = sin(X * np.pi) * 2
    else:
        raise NotImplementedError

    assert np.allclose(P.eval_f(P.u_init), 0), 'Non-zero time derivative in static 0 configuration'

    u = P.u_init
    for i in [iu, iv, iT, ip]:
        u[i][:] = y
    u[ivz] = y_z
    u[iTz] = y_z
    u[iuz] = y_z

    f = P.eval_f(u, compute_violations=compute_violations)

    for i in [ivz, iTz, iuz]:
        assert np.allclose(f.impl[i] + f.expl[i], 0), f'Non-zero time derivative in algebraic component {i}'

    f_expect = P.f_init
    f_expect.expl[iT] = -y * (y_x + y_z)
    f_expect.impl[iT] = kappa * (y_xx + y_zz)
    f_expect.expl[iu] = -y * y_z - y * y_x
    f_expect.impl[iu] = -y_x + nu * (y_xx + y_zz)
    f_expect.expl[iv] = -y * (y_z + y_x)
    f_expect.impl[iv] = -y_z + nu * (y_xx + y_zz) + y

    for comp in ['u', 'v', 'T']:
        i = P.spectral.index(comp)
        assert np.allclose(f.impl[i], f_expect.impl[i]), f'Unexpected implicit function evaluation in component {comp}'
        assert np.allclose(f.expl[i], f_expect.expl[i]), f'Unexpected explicit function evaluation in component {comp}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [1, 2**7])
@pytest.mark.parametrize('nz', [2**5])
@pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('T_top', [2])
@pytest.mark.parametrize('T_bottom', [3.14, -9])
@pytest.mark.parametrize('v_top', [2.77])
@pytest.mark.parametrize('noise', [0, 1e-3])
def test_BCs(nx, nz, cheby_mode, T_top, T_bottom, v_top, noise, plotting=False):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'T_top': T_top,
        'T_bottom': T_bottom,
        'v_top': v_top,
        'v_bottom': v_top,
        'u_top': 0,
        'u_bottom': 0,
        'p_integral': 0,
    }
    P = RayleighBenard(
        nx=nx, nz=nz, cheby_mode=cheby_mode, BCs=BCs, right_preconditioning='T2T', left_preconditioner=False
    )

    u = P.u_exact(0, noise_level=noise, kzmax=None, kxmax=None)
    u = P.solve_system(u, 1e0)

    P.check_BCs(u)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [4])
@pytest.mark.parametrize('cheby_mode', ['T2U'])
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
def test_vorticity(nx, nz, cheby_mode, direction):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    assert nz > 3
    assert nx > 8

    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode)
    iu, iv = P.index(['u', 'v'])

    u = P.u_init

    if direction == 'x':
        u[iv] = np.sin(P.X * np.pi)
        u[iu] = np.cos(P.X * np.pi)
        expect = np.cos(P.X * np.pi) * np.pi
    elif direction == 'z':
        u[iv] = P.Z**2
        u[iu] = P.Z**3
        expect = 3 * P.Z**2
    elif direction == 'mixed':
        u[iv] = np.sin(P.X * np.pi) * P.Z**2
        u[iu] = np.cos(P.X * np.pi) * P.Z**3
        expect = np.cos(P.X * np.pi) * np.pi * P.Z**2 + np.cos(P.X * np.pi) * 3 * P.Z**2
    else:
        raise NotImplementedError

    assert np.allclose(P.compute_vorticity(u), expect)


# @pytest.mark.mpi4py
# @pytest.mark.parametrize('nx', [32])
# @pytest.mark.parametrize('nz', [32])
# @pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
# @pytest.mark.parametrize('direction', ['x', 'mixed'])
# def test_linear_operator(nx, nz, cheby_mode, direction):
#     import numpy as np
#     from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
#     import matplotlib.pyplot as plt
#     import scipy.sparse as sp
#
#     P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode)
#
#     u = P.u_init
#     expect = P.u_init
#
#     conv = sp.kron(
#         sp.eye(u.shape[0]), sp.kron(P.spectral.axes[0].get_Id(), P.spectral.axes[1].get_conv(cheby_mode[::-1]))
#     )
#
#     for i in [P.iu, P.iv, P.iT, P.ip]:
#         if direction == 'x':
#             u[i] = np.sin(P.X * (i + 1))
#         elif direction == 'mixed':
#             u[i] = P.Z**2 * np.sin(P.X)
#         else:
#             raise NotImplementedError
#
#     derivatives = P.compute_z_derivatives(u)
#     for i in [P.iTz, P.ivz]:
#         u[i] = derivatives[i]
#
#     if direction == 'x':
#         expect[P.iu] = P.Pr * (-(P.ip + 1) * np.cos(P.X * (P.ip + 1)) - (P.iu + 1) ** 2 * np.sin(P.X * (P.iu + 1)))
#         expect[P.iv] = P.Pr * P.Ra * u[P.iT]
#         expect[P.ivz] = -(P.iu + 1) * np.cos(P.X * (P.iu + 1))
#         expect[P.iT] = -((P.iT + 1) ** 2) * np.sin(P.X * (P.iT + 1))
#         expect[P.iTz] = 0
#         expect[P.ip] = np.sin(P.X * (P.ip + 1)) * P.Z
#         expect[P.ip] = -np.cos(P.X * (P.ip + 1)) / (P.ip + 1) * P.Z
#     elif direction == 'mixed':
#         expect[P.iu] = P.Pr * (-np.cos(P.X) - np.sin(P.X)) * P.Z**2
#         expect[P.iv] = P.Pr * (-np.sin(P.X) * 2 * P.Z + P.Ra * u[P.iT] + 2 * np.sin(P.X))
#         # expect[P.ivz] = u[P.ivz] - u[P.iux]
#         expect[P.iT] = 2 * np.sin(P.X) - np.sin(P.X) * P.Z**2
#         expect[P.iTz] = 0
#         expect[P.ip] = -np.cos(P.X) * P.Z**3 / 3
#     else:
#         raise NotImplementedError
#
#     u_hat = P.transform(u)
#     Lu_hat = (conv @ P.L @ u_hat.flatten()).reshape(u.shape)
#     Lu = P.itransform(Lu_hat)
#
#     fig, axs = plt.subplots(1, 4)
#     i = P.iT
#     im = axs[0].pcolormesh(P.X, P.Z, u[i].real)
#     im = axs[1].pcolormesh(P.X, P.Z, Lu[i].real)
#     im = axs[2].pcolormesh(P.X, P.Z, expect[i].real)
#     im = axs[3].pcolormesh(P.X, P.Z, (Lu[i] - expect[i]).real)
#     fig.colorbar(im)
#     # plt.show()
#
#     for i in range(u.shape[0]):
#         assert np.allclose(Lu[i], expect[i]), f'Got unexpected result in component {P.index_to_name[i]}'


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [32])
@pytest.mark.parametrize('nz', [32])
@pytest.mark.parametrize('T_top', [2])
@pytest.mark.parametrize('T_bottom', [3.14])
@pytest.mark.parametrize('v_top', [2.77])
@pytest.mark.parametrize('v_bottom', [2.77])
def test_initial_conditions(nx, nz, T_top, T_bottom, v_top, v_bottom):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'T_top': T_top,
        'T_bottom': T_bottom,
        'v_top': v_top,
        'v_bottom': v_bottom,
        'p_integral': 0,
    }

    P = RayleighBenard(nx=nx, nz=nz, BCs=BCs)
    u0 = P.u_exact()
    P.spectral.check_BCs(u0)


# @pytest.mark.mpi4py
# @pytest.mark.parametrize('limit', ['Ra->0', 'Pr->0', 'Pr->inf'])
# def test_limit_case(limit, nx=2**6, nz=2**5, plotting=False):
#     import numpy as np
#     from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
#
#     if limit == 'Ra->0':
#         P = RayleighBenard(nx=nx, nz=nz, Rayleigh=1e-8, Prandl=1)
#         rhs = P.u_exact(0, 1e1)
#         sol = P.solve_system(rhs, 1e7)
#         expect = P.u_exact(0, 0)
#         thresh = 1e-10
#
#     elif limit == 'Pr->0':
#         P = RayleighBenard(nx=nx, nz=nz, Rayleigh=1e0, Prandl=1e-8)
#         rhs = P.u_exact(0, 1e1)
#         sol = P.solve_system(rhs, 1e9)
#         expect = P.u_exact(0, 0)
#         thresh = 1e-10
#
#     elif limit == 'Pr->inf':
#         P = RayleighBenard(nx=nx, nz=nz, Rayleigh=1e0, Prandl=1e20)
#
#         rhs = P.u_exact(0, 1e-2)
#         sol = P.solve_system(rhs.copy(), 1e0)
#
#         expect = P.u_exact(0, 0)
#
#         derivatives = P.u_init
#         idxp, idzp = 0, 1
#         sol_hat = P.transform(sol)
#         derivatives[idxp] = (P.U2T @ P.Dx @ sol_hat[P.index('p')].flatten()).reshape(derivatives[idxp].shape)
#         derivatives[idzp] = (P.U2T @ P.Dz @ sol_hat[P.index('p')].flatten()).reshape(derivatives[idzp].shape)
#         derivatives = P.itransform(derivatives)
#
#         P.plot(derivatives, quantity='v')
#         P.plot(sol, quantity='p')
#         import matplotlib.pyplot as plt
#
#         # plt.show()
#
#         # assert np.allclose(derivatives[idxp], 0), 'Got non-zero pressure derivative in x-direction'
#         # assert np.allclose(derivatives[idzp], sol[P.index('T')]), 'Got unexpected pressure derivative in z-direction'
#
#         for component in ['T', 'Tz', 'p']:
#             expect[P.index(component)] = rhs[P.index(component)]
#         thresh = (P.Rayleigh * P.Prandl) ** (-1 / 2.0) * 10.0
#
#     else:
#         raise NotImplementedError
#
#     def compute_errors(u1, u2, msg, thresh=1e-10, components=P.components):
#         msgs = ''
#         for comp in components:
#             i = P.index(comp)
#             error = abs(u1[i] - u2[i])
#             if error > thresh:
#                 msgs = f'{msgs} {error=:2e} in {comp}'
#         if plotting and msgs != '':
#             print(f'Errors too large in limit {msg}: {msgs}')
#         else:
#             assert msgs == '', f'Errors too large when solving {msg}: {msgs}'
#
#         violations = P.compute_constraint_violation(u2)
#         for key in [key for key in violations.keys() if key in components]:
#             if plotting and abs(violations[key]) > 1e-12:
#                 print(f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!')
#             else:
#                 assert np.allclose(
#                     violations[key], 0
#                 ), f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!'
#
#     compute_errors(sol, expect, limit, thresh=thresh, components = ['T', 'u', 'v', 'uz', 'vz', 'Tz'])
#     if plotting:
#         import matplotlib.pyplot as plt
#
#         q = 'p'
#         P.plot(expect, quantity=q)
#         P.plot(sol, quantity=q)
#         plt.show()
#
#
# @pytest.mark.mpi4py
# def test_solver_small_step_size(plotting=False):
#     import numpy as np
#     from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
#     import matplotlib.pyplot as plt
#
#     try:
#         from mpi4py import MPI
#
#         comm = MPI.COMM_WORLD
#     except ModuleNotFoundError:
#         comm = None
#
#     P = RayleighBenard(
#         nx=2**4,
#         nz=2**6,
#         cheby_mode='T2U',
#         comm=comm,
#         solver_type='direct',
#         left_preconditioner=True,
#         right_preconditioning='D2T',
#         Rayleigh=1.0,
#     )
#
#     u0 = P.u_exact(noise_level=1e-3, kzmax=4, kxmax=4)
#     u_solver = P.solve_system(u0, 1e-4)
#
#     u_hat = P.transform(u_solver)
#     print(np.max(P.L @ u_hat.flatten()))
#
#     # generate stationary solution without mass matrix
#     A = P.put_BCs_in_matrix(P.L)
#     rhs = P.transform(P.put_BCs_in_rhs(P.u_init))
#     sol = P.itransform(P.sparse_lib.linalg.spsolve(A, rhs.flatten()).reshape(rhs.shape))
#
#     u_solver = P.solve_system(sol, dt=1e0)
#
#     compute_errors(u_solver, u0, 'tiny step size', 1e-9, raise_errors=not plotting, P=P)
#
#     if plotting:
#         P.plot(sol, quantity='Tz')
#         P.plot(u_solver, quantity='Tz')
#         plt.show()
#
#
# @pytest.mark.mpi4py
# @pytest.mark.parametrize('nx', [32])
# @pytest.mark.parametrize('nz', [32])
# @pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
# @pytest.mark.parametrize('noise', [1e-3, 0])
# def test_solver(nx, nz, cheby_mode, noise, plotting=False):
#     import numpy as np
#     from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard, CFLLimit
#     import matplotlib.pyplot as plt
#
#     try:
#         from mpi4py import MPI
#
#         comm = MPI.COMM_WORLD
#     except ModuleNotFoundError:
#         comm = None
#
#     P = RayleighBenard(
#         nx=nx,
#         nz=nz,
#         cheby_mode=cheby_mode,
#         comm=comm,
#         # left_preconditioner=False,
#         # right_preconditioning='T2T',
#         Rayleigh=2e6 / 16,
#     )
#
#     def IMEX_Euler(_u, dt):
#         f = P.eval_f(_u)
#         un = P.solve_system(_u + dt * f.expl, dt)
#         return un
#
#     def compute_errors(u1, u2, msg, thresh=1e-10, components=['u', 'v', 'T']):
#         msgs = ''
#         for comp in components:
#             i = P.index(comp)
#             error = abs(u1[i] - u2[i])
#             if error > thresh:
#                 msgs = f'{msgs} {error=:2e} in {comp}'
#             if not (np.allclose(u2[i].imag, 0) and np.allclose(u1[i].imag, 0)):
#                 msgs = f'{msgs} non-zero imaginary part in {comp}'
#         if plotting and msgs != '':
#             print(f'Errors too large when solving {msg}: {msgs}')
#         else:
#             assert msgs == '', f'Errors too large when solving {msg}: {msgs}'
#
#         violations = P.compute_constraint_violation(u2)
#         for key in [key for key in violations.keys() if key in components]:
#             if plotting and abs(violations[key]) > 1e-11:
#                 print(f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!')
#             else:
#                 assert np.allclose(
#                     violations[key], 0
#                 ), f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!'
#
#
#     dt = 1e-3
#     u0 = P.solve_system(P.u_exact(noise_level=noise), dt=1000)
#
#
#     f0 = P.eval_f(u0)
#     P.plot(u0)
#     import matplotlib.pyplot as plt
#     # plt.show()
#     dt = 0.4 * CFLLimit.compute_max_step_size(P, u0)
#
#     un_no_convection = P.solve_system(u0, dt)
#     u02_no_convection = un_no_convection - dt * P.eval_f(un_no_convection).impl
#     compute_errors(u0, u02_no_convection, 'without convection')
#
#     un = P.solve_system(u0 + dt * f0.expl, dt)
#     fn = P.eval_f(un, compute_violations=True)
#     violations = [abs(f0.impl[P.index(i)]) for i in ['uz', 'vz', 'Tz']]
#     print(violations)
#     u02 = un - dt * fn.impl - dt * f0.expl
#     compute_errors(u0, u02, 'with convection')
#     return None
#
#     P_heat = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode, Rayleigh=1e-8)
#     poisson = P_heat.solve_system(P_heat.u_exact(0, 1e1), 1e7)
#     expect_poisson = P_heat.u_exact(0, 0)
#     compute_errors(poisson, expect_poisson, 'Poisson', components=['u', 'v', 'T', 'Tz', 'vz', 'uz'])
#
#     u_static = P.u_exact(noise_level=0)
#     static = P.solve_system(u_static, 1e-0)
#     compute_errors(u_static, static, 'static configuration', components=['u', 'v', 'T', 'Tz', 'vz', 'uz'])
#
#     # get a relaxed initial condition by smoothing the noise using a large implicit Euler step
#     if noise == 0:
#         u0 = P.u_exact()
#     else:
#         _u0 = P.u_exact(noise_level=noise)
#         u0 = P.solve_system(_u0, 0.25)
#
#     small_dt = P.solve_system(u0, 1e0)
#     compute_errors(u0, small_dt, 'tiny step size', 1e-9, components=['u', 'v', 'T', 'uz', 'vz', 'Tz'])
#     # return None
#
#     u0 = P.u_exact(noise_level=noise)
#     # print('T0', P.transform(u0)[P.iT])
#     dt = 1e-1
#     forward = P.solve_system(u0, dt)
#     # print('T ', P.transform(forward)[P.iT])
#     # print('Tz', P.transform(forward)[P.iTz])
#     # print('Yz', (P.Dz @ P.transform(forward)[P.iT].flatten()).reshape(u0[P.iT].shape) - P.transform(forward)[P.iTz])
#     f = P.eval_f(forward)
#     backward = forward - dt * (f.impl)
#     compute_errors(u0, backward, 'backward without convection', 1e-8, components=['T', 'u', 'v'])
#     # P.plot(forward, quantity='vz')
#     # plt.show()
#     # return None
#
#     # dt = 1e-1
#     # forward = IMEX_Euler(u0, dt)
#     # f = P.eval_f(forward)
#     # f_before = P.eval_f(u0)
#     # backward = forward - dt * (f.impl + f_before.expl)
#     # compute_errors(u0, backward, 'backward', 1e-6, components=['T', 'u', 'v'])
#     # return None
#
#     if plotting:
#         u = P.u_exact(noise_level=1e-3)
#         t = 0
#         nsteps = 1000
#         dt = 0.25
#         fig = P.get_fig()
#         # P.plot(u, t, fig=fig, quantity='u')
#         # plt.show()
#         # return None
#         print('hi')
#
#         for i in range(nsteps):
#             t += dt
#             u = IMEX_Euler(u, dt)
#             # f = P.eval_f(u)
#
#             P.plot(u, t, fig=fig, quantity='T')
#             plt.pause(1e-8)
#         # plt.show()


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [1, 8])
@pytest.mark.parametrize('component', ['u', 'T'])
def test_Possion_problems(nx, component):
    """
    When forgetting about convection and the time-dependent part, you get Poisson problems in u and T that are easy to solve. We check that we get the exact solution in a simple test here.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'T_top': 0,
        'T_bottom': 0,
    }
    P = RayleighBenard(nx=nx, nz=6, BCs=BCs, cheby_mode='T2T', Rayleigh=1.0)
    rhs = P.u_init

    idx = P.index(f'{component}')
    idx_z = P.index(f'{component}z')

    A = P.put_BCs_in_matrix(-P.L)
    rhs[idx][0, 2] = 6
    rhs[idx][0, 0] = 6
    u = P.sparse_lib.linalg.spsolve(A, P.M @ rhs.flatten()).reshape(rhs.shape).real

    u_exact = P.u_init
    u_exact[idx][0, 4] = 1 / 8
    u_exact[idx][0, 2] = 1 / 2
    u_exact[idx][0, 0] = -5 / 8
    u_exact[idx_z][0, 3] = 1
    u_exact[idx_z][0, 1] = 3

    if component == 'T':
        ip = P.index('p')
        u_exact[ip][0, 5] = 1 / (16 * 5)
        u_exact[ip][0, 3] = 5 / (16 * 5)
        u_exact[ip][0, 1] = -70 / (16 * 5)

    assert np.allclose(u_exact, u)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nz', [4, 6])
@pytest.mark.parametrize('nx', [1, 4, 5])
def test_resolution_derefinement(nz, nx):
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'T_top': 0,
        'T_bottom': 0,
    }
    P = RayleighBenard(nx=nx, nz=nz, BCs=BCs, Rayleigh=1)

    remove_modes = 1

    component = 'u'
    idx = P.index(f'{component}')

    def solve(prob):
        A = prob.put_BCs_in_matrix(-prob.L)

        rhs = prob.u_init
        rhs = prob.put_BCs_in_rhs(rhs)
        rhs[idx][0, 2] = 6
        rhs[idx][0, 0] = 6
        return prob.sparse_lib.linalg.spsolve(A, rhs.flatten()).reshape(rhs.shape).real

    u_hat = solve(P)
    assert P.check_derefinement_ok(u_hat, remove_modes=1) == (nz > 4)

    while P.check_derefinement_ok(u_hat, remove_modes=1):
        u_hat, P = P.derefine_resolution(u_hat, remove_modes=1)
        u_hat = solve(P)

        assert u_hat.shape[2] < 6
        assert u_hat.shape[2] >= 4


@pytest.mark.mpi4py
@pytest.mark.parametrize('nz', [4, 6])
@pytest.mark.parametrize('nx', [1, 4, 5])
def test_resolution_refinement(nz, nx):
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'T_top': 0,
        'T_bottom': 0,
    }
    P = RayleighBenard(nx=nx, nz=nz, BCs=BCs, Rayleigh=1)

    component = 'u'
    idx = P.index(f'{component}')

    def solve(prob):
        A = prob.put_BCs_in_matrix(-prob.L)

        rhs = prob.u_init
        rhs = prob.put_BCs_in_rhs(rhs)
        rhs[idx][0, 2] = 6
        rhs[idx][0, 0] = 6
        return prob.sparse_lib.linalg.spsolve(A, rhs.flatten()).reshape(rhs.shape).real

    u_hat = solve(P)
    assert P.check_refinement_needed(u_hat) == (nz <= 4)

    while P.check_refinement_needed(u_hat):
        _u_hat, P = P.refine_resolution(u_hat)
        u_hat = solve(P)

        assert u_hat.shape[2] <= 6
        assert u_hat.shape[2] > 4


@pytest.mark.mpi4py
@pytest.mark.parametrize('nz', [4, 5])
def test_refinement_2D(nz):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    P = RayleighBenard(nx=2 * nz, nz=nz)

    u_hat = P.u_init_forward
    u_hat[...] = np.random.random(u_hat.shape)
    # u_hat[0][:] = np.arange(nz)

    u_refined, P_refined = P.refine_resolution2(u_hat, factor=2)
    u_hat_refined = P_refined.transform(u_refined)
    print(u_hat[0])
    print(u_hat_refined[0].real)

    u_pad = P.itransform(u_hat, padding=[2, 2]).real
    print(P_refined.transform(u_pad)[0].real)
    u_refined = P_refined.itransform(u_hat_refined)
    # print(u_pad[0])
    # print(u_refined[0])

    assert np.allclose(u_pad, u_refined)

    # u_hat_derefined, P_derefined = P_refined.derefine_resolution(u_hat_refined, remove_modes=nz)
    u_hat_derefined, P_derefined = P_refined.derefine_resolution2(u_refined, factor=2)
    print(u_hat_derefined[0])

    assert np.allclose(u_hat, u_hat_derefined)
    # print(u_hat, u_hat_


@pytest.mark.mpi4py
def test_Poisson_problem_v():
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'u_top': 0,
        'u_bottom': 0,
        'v_top': 0,
        'v_bottom': 0,
        'T_top': 0,
        'T_bottom': 2,
    }
    P = RayleighBenard(nx=1, nz=8, BCs=BCs, Rayleigh=1.0)
    rhs = P.u_init

    idx = P.index('v')

    A = P.put_BCs_in_matrix(-P.L)
    rhs_real = P.put_BCs_in_rhs(rhs)
    rhs = P.transform(rhs_real)
    # rhs[idx][0, 2] = 6
    # rhs[idx][0, 0] = 6
    u = P.sparse_lib.linalg.spsolve(A, P.M @ rhs.flatten()).reshape(rhs.shape).real

    u_exact = P.u_init
    iT = P.index('T')
    u_exact[iT][0, 1] = 1
    u_exact[iT][0, 0] = 1

    ip = P.index('p')
    u_exact[ip][0, 5] = 1 / (16 * 5)
    u_exact[ip][0, 3] = 5 / (16 * 5)
    u_exact[ip][0, 1] = -70 / (16 * 5)

    print(u[iT])
    # print(u_exact[ip])
    assert np.allclose(u_exact, u)


if __name__ == '__main__':
    # test_limit_case('Pr->inf', plotting=True)
    # test_derivatives(64, 64, 'z', 'T2U')
    # test_eval_f(1, 8, 'T2T', 'z', True)
    # test_BCs(2**8, 2**6 + 0, 'T2U', 2.77, 3.14, 2, 0.001, True)
    # test_solver(2**7, 2**5 + 0, 'T2U', noise=0e-3, plotting=True)
    # test_solver(2**8, 2**6 + 0, 'T2U', noise=1e3, plotting=True)
    # test_resolution_derefinement(6, 3)
    # test_resolution_refinement(4, 4)
    # test_refinement_2D(2)
    test_Poisson_problem_v()
    # test_solver_small_step_size(True)
    # test_vorticity(64, 4, 'T2T', 'x')
    # test_linear_operator(2**4, 2**4, 'T2U', 'x')
    # test_initial_conditions(4, 5, 0, 1, 1, 1)
    print('done')
