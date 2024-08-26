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


# def test_RayleighBenard():
#     import numpy as np
#     from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
#
#     P = RayleighBenard(**PARAMS)
#
#     u0 = P.u_exact()
#
#     f = P.eval_f(u0)
#
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(1, 2)
#     i = P.iTz
#     im = axs[0].pcolormesh(P.X, P.Z, u0[i].real)
#     axs[1].pcolormesh(P.X, P.Z, (f.impl + f.expl)[i].real)
#     fig.colorbar(im)
#     plt.show()
#
#     assert np.allclose(P.eval_f(P.u_init), 0), 'Non-zero time derivative in static 0 configuration'
#     return None
#
#     # u = P.u_exact(0)
#     u_sin = P.u_init
#     for i in range(u_sin.shape[0]):
#         u_sin[i] = (np.cos(P.X[0]/P.L[0]*2.0*np.pi)+np.sin(P.X[1]/P.L[1]*2.0*np.pi) - 1) * np.exp(-(P.X[1] - np.pi)**2/4)
#
#     f = P.eval_f(u_sin, 0)
#
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(2, 2, figsize=(9, 4))
#     i = P.iUz
#     axs[0,0].pcolormesh(P.X[0], P.X[1], u_sin[P.iUx])
#     axs[0,1].pcolormesh(P.X[0], P.X[1], f.impl[P.iUx])
#
#     axs[1,0].pcolormesh(P.X[0], P.X[1],u_sin[P.iUz])
#     axs[1,1].pcolormesh(P.X[0], P.X[1],f.impl[P.iUz])
#     print(f.impl[P.iUz])
#     plt.show()


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
def test_eval_f(nx, nz, cheby_mode, direction):
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
    u[P.index('uz')] = y_z

    f = P.eval_f(u, compute_violations=False)

    for i in [ivz, iTz]:
        assert np.allclose(f.impl[i] + f.expl[i], 0), f'Non-zero time derivative in algebraic component {i}'

    f_expect = P.f_init
    f_expect.expl[iT] = -y * (y_x + y_z)
    f_expect.impl[iT] = kappa * (y_xx + y_zz)
    f_expect.expl[iu] = -y * y_z - y * y_x
    f_expect.impl[iu] = -y_x + nu * (y_xx + y_zz)
    f_expect.expl[iv] = -y * (y_z + y_x)
    f_expect.impl[iv] = -y_z + nu * (y_xx + y_zz) + y

    for comp in P.spectral.components[::-1]:
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


@pytest.mark.mpi4py
@pytest.mark.parametrize('limit', ['Ra->0', 'Pr->0', 'Pr->inf'])
def test_limit_case(limit, nx=2**6, nz=2**5, plotting=False):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    if limit == 'Ra->0':
        P = RayleighBenard(nx=nx, nz=nz, Rayleigh=1e-8, Prandl=1)
        rhs = P.u_exact(0, 1e1)
        sol = P.solve_system(rhs, 1e7)
        expect = P.u_exact(0, 0)
        thresh = 1e-10

    elif limit == 'Pr->0':
        P = RayleighBenard(nx=nx, nz=nz, Rayleigh=1e0, Prandl=1e-8)
        rhs = P.u_exact(0, 1e1)
        sol = P.solve_system(rhs, 1e9)
        expect = P.u_exact(0, 0)
        thresh = 1e-10

    elif limit == 'Pr->inf':
        P = RayleighBenard(nx=nx, nz=nz, Rayleigh=1e0, Prandl=1e20)

        rhs = P.u_exact(0, 1e-2)
        sol = P.solve_system(rhs.copy(), 1e0)

        expect = P.u_exact(0, 0)

        derivatives = P.u_init
        idxp, idzp = 0, 1
        sol_hat = P.transform(sol)
        derivatives[idxp] = (P.U2T @ P.Dx @ sol_hat[P.index('p')].flatten()).reshape(derivatives[idxp].shape)
        derivatives[idzp] = (P.U2T @ P.Dz @ sol_hat[P.index('p')].flatten()).reshape(derivatives[idzp].shape)
        derivatives = P.itransform(derivatives)

        P.plot(derivatives, quantity='v')
        P.plot(sol, quantity='p')
        import matplotlib.pyplot as plt

        # plt.show()

        # assert np.allclose(derivatives[idxp], 0), 'Got non-zero pressure derivative in x-direction'
        assert np.allclose(derivatives[idzp], sol[P.index('T')]), 'Got unexpected pressure derivative in z-direction'

        for component in ['T', 'Tz', 'p']:
            expect[P.index(component)] = rhs[P.index(component)]
        thresh = (P.Rayleigh * P.Prandl) ** (-1 / 2.0) * 10.0

    else:
        raise NotImplementedError

    def compute_errors(u1, u2, msg, thresh=1e-10, components=P.components):
        msgs = ''
        for comp in components:
            i = P.index(comp)
            error = abs(u1[i] - u2[i])
            if error > thresh:
                msgs = f'{msgs} {error=:2e} in {comp}'
        if plotting and msgs != '':
            print(f'Errors too large in limit {msg}: {msgs}')
        else:
            assert msgs == '', f'Errors too large when solving {msg}: {msgs}'

        violations = P.compute_constraint_violation(u2)
        for key in [key for key in violations.keys() if key in components]:
            if plotting and abs(violations[key]) > 1e-12:
                print(f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!')
            else:
                assert np.allclose(
                    violations[key], 0
                ), f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!'

    compute_errors(sol, expect, limit, thresh=thresh)
    if plotting:
        import matplotlib.pyplot as plt

        q = 'p'
        P.plot(expect, quantity=q)
        P.plot(sol, quantity=q)
        plt.show()


@pytest.mark.mpi4py
def test_solver_small_step_size(plotting=False):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
    import matplotlib.pyplot as plt

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    except ModuleNotFoundError:
        comm = None

    P = RayleighBenard(
        nx=2**4,
        nz=2**6,
        cheby_mode='T2U',
        comm=comm,
        solver_type='direct',
        left_preconditioner=True,
        right_preconditioning='D2T',
        Rayleigh=1.0,
    )

    u0 = P.u_exact(noise_level=1e-3, kzmax=4, kxmax=4)
    u_solver = P.solve_system(u0, 1e-4)

    u_hat = P.transform(u_solver)
    print(np.max(P.L @ u_hat.flatten()))

    # generate stationary solution without mass matrix
    A = P.put_BCs_in_matrix(P.L)
    rhs = P.transform(P.put_BCs_in_rhs(P.u_init))
    sol = P.itransform(P.sparse_lib.linalg.spsolve(A, rhs.flatten()).reshape(rhs.shape))

    u_solver = P.solve_system(sol, dt=1e0)

    compute_errors(u_solver, u0, 'tiny step size', 1e-9, raise_errors=not plotting, P=P)

    if plotting:
        P.plot(sol, quantity='Tz')
        P.plot(u_solver, quantity='Tz')
        plt.show()


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [32])
@pytest.mark.parametrize('nz', [32])
@pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('noise', [1e-3, 0])
def test_solver(nx, nz, cheby_mode, noise, plotting=False):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
    import matplotlib.pyplot as plt

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    except ModuleNotFoundError:
        comm = None

    P = RayleighBenard(
        nx=nx,
        nz=nz,
        cheby_mode=cheby_mode,
        comm=comm,
        solver_type='direct',
        left_preconditioner=False,
        right_preconditioning='T2T',
        Rayleigh=2e6 / 8,
        # Rayleigh=1,
    )

    def IMEX_Euler(_u, dt):
        f = P.eval_f(_u)
        un = P.solve_system(_u + dt * f.expl, dt)
        return un

    def compute_errors(u1, u2, msg, thresh=1e-10, components=P.components):
        msgs = ''
        for comp in components:
            i = P.index(comp)
            error = abs(u1[i] - u2[i])
            if error > thresh:
                msgs = f'{msgs} {error=:2e} in {comp}'
            if not (np.allclose(u2[i].imag, 0) and np.allclose(u1[i].imag, 0)):
                msgs = f'{msgs} non-zero imaginary part in {comp}'
        if plotting and msgs != '':
            print(f'Errors too large when solving {msg}: {msgs}')
        else:
            assert msgs == '', f'Errors too large when solving {msg}: {msgs}'

        violations = P.compute_constraint_violation(u2)
        for key in [key for key in violations.keys() if key in components]:
            if plotting and abs(violations[key]) > 1e-11:
                print(f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!')
            else:
                assert np.allclose(
                    violations[key], 0
                ), f'Violation of constraints in {key}: {abs(violations[key]):.2e} after solving {msg}!'

    P_heat = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode, Rayleigh=1e-8)
    poisson = P_heat.solve_system(P_heat.u_exact(0, 1e1), 1e7)
    expect_poisson = P_heat.u_exact(0, 0)
    compute_errors(poisson, expect_poisson, 'Poisson', components=['u', 'v', 'T', 'Tz', 'vz', 'uz'])

    u_static = P.u_exact(noise_level=0)
    static = P.solve_system(u_static, 1e-0)
    compute_errors(u_static, static, 'static configuration', components=['u', 'v', 'T', 'Tz', 'vz', 'uz'])

    # get a relaxed initial condition by smoothing the noise using a large implicit Euler step
    if noise == 0:
        u0 = P.u_exact()
    else:
        _u0 = P.u_exact(noise_level=noise)
        u0 = P.solve_system(_u0, 0.25)

    small_dt = P.solve_system(u0, 1e0)
    compute_errors(u0, small_dt, 'tiny step size', 1e-9, components=['u', 'v', 'T', 'uz', 'vz', 'Tz'])
    # return None

    u0 = P.u_exact(noise_level=noise)
    # print('T0', P.transform(u0)[P.iT])
    dt = 1e-1
    forward = P.solve_system(u0, dt)
    # print('T ', P.transform(forward)[P.iT])
    # print('Tz', P.transform(forward)[P.iTz])
    # print('Yz', (P.Dz @ P.transform(forward)[P.iT].flatten()).reshape(u0[P.iT].shape) - P.transform(forward)[P.iTz])
    f = P.eval_f(forward)
    backward = forward - dt * (f.impl)
    compute_errors(u0, backward, 'backward without convection', 1e-8, components=['T', 'u', 'v'])
    # P.plot(forward, quantity='vz')
    # plt.show()
    # return None

    # dt = 1e-1
    # forward = IMEX_Euler(u0, dt)
    # f = P.eval_f(forward)
    # f_before = P.eval_f(u0)
    # backward = forward - dt * (f.impl + f_before.expl)
    # compute_errors(u0, backward, 'backward', 1e-6, components=['T', 'u', 'v'])
    # return None

    if plotting:
        u = P.u_exact(noise_level=1e-3)
        t = 0
        nsteps = 1000
        dt = 0.25
        fig = P.get_fig()
        # P.plot(u, t, fig=fig, quantity='u')
        # plt.show()
        # return None
        print('hi')

        for i in range(nsteps):
            t += dt
            u = IMEX_Euler(u, dt)
            # f = P.eval_f(u)

            P.plot(u, t, fig=fig, quantity='T')
            plt.pause(1e-8)
        # plt.show()


if __name__ == '__main__':
    # test_limit_case('Pr->inf', plotting=True)
    # test_derivatives(64, 64, 'z', 'T2U')
    # test_eval_f(16, 8, 'T2U', 'z')
    test_BCs(2**8, 2**6 + 0, 'T2U', 2.77, 3.14, 2, 0.001, True)
    # test_solver(2**7, 2**5 + 0, 'T2U', noise=0e-3, plotting=True)
    # test_solver(2**1, 2**1 + 0, 'T2U', noise=0e-3, plotting=True)
    # test_solver_small_step_size(True)
    # test_vorticity(64, 4, 'T2T', 'x')
    # test_linear_operator(2**4, 2**4, 'T2U', 'x')
    # test_initial_conditions(4, 5, 0, 1, 1, 1)
    print('done')
