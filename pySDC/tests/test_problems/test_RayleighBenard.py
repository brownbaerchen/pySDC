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


@pytest.mark.base
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


@pytest.mark.base
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
@pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
def test_eval_f(nx, nz, cheby_mode, direction):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode)
    X, Z = P.X, P.Z
    cos, sin = np.cos, np.sin

    kappa = (P.Rayleigh * P.Prandl) ** (-1 / 2)
    nu = (P.Rayleigh / P.Prandl) ** (-1 / 2)

    if direction == 'x':
        y = sin(X)
        y_x = cos(X)
        y_xx = -sin(X)
        y_z = 0
        y_zz = 0
    elif direction == 'z':
        y = Z**2
        y_x = 0
        y_xx = 0
        y_z = 2 * Z
        y_zz = 2.0
    elif direction == 'mixed':
        y = sin(X) * Z**2
        y_x = cos(X) * Z**2
        y_xx = -sin(X) * Z**2
        y_z = sin(X) * 2 * Z
        y_zz = sin(X) * 2
    else:
        raise NotImplementedError

    assert np.allclose(P.eval_f(P.u_init), 0), 'Non-zero time derivative in static 0 configuration'

    u = P.u_init
    for i in [P.iu, P.iv, P.iT, P.ip]:
        u[i][:] = y
    u[P.ivz] = y_z
    u[P.iTz] = y_z
    u[P.index('uz')] = y_z

    f = P.eval_f(u)

    for i in [P.ivz, P.iTz]:
        assert np.allclose(f.impl[i] + f.expl[i], 0), f'Non-zero time derivative in algebraic component {i}'

    f_expect = P.f_init
    f_expect.expl[P.iT] = -y * (y_x + y_z)
    f_expect.impl[P.iT] = kappa * (y_xx + y_zz)
    f_expect.expl[P.iu] = -y * y_z - y * y_x
    f_expect.impl[P.iu] = -y_x + nu * y_xx
    f_expect.expl[P.iv] = -y * (y_z + y_x)
    f_expect.impl[P.iv] = -y_z + nu * y_zz + y

    for comp in P.spectral.components:
        i = P.spectral.index(comp)
        assert np.allclose(f.impl[i], f_expect.impl[i]), f'Unexpected implicit function evaluation in component {comp}'
        assert np.allclose(f.expl[i], f_expect.expl[i]), f'Unexpected explicit function evaluation in component {comp}'


@pytest.mark.base
@pytest.mark.parametrize('nx', [1, 4])
@pytest.mark.parametrize('nz', [32])
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
        'p_integral': 0,
    }
    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode, BCs=BCs)

    rhs = P.u_exact(0, noise_level=noise)
    sol = P.solve_system(rhs, 1e0)

    expect = {}
    for q in ['T', 'v']:
        if nz == 2 or True:
            expect[q] = (BCs[f'{q}_top'] - BCs[f'{q}_bottom']) / 2 * P.Z + (BCs[f'{q}_top'] + BCs[f'{q}_bottom']) / 2
        else:
            raise NotImplementedError

    sol_hat = P.transform(sol, axes=(1,))
    pressure_coef = sol_hat[P.ip][0]
    pressure_integral = np.polynomial.Chebyshev(pressure_coef).integ(1, lbnd=-1)(1)

    zero = np.zeros_like(expect['T'])

    if plotting:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(np.log(abs(_A.toarray())))
        # axs[1].imshow(np.log(abs(A.toarray())))
        # plt.show()
        # for i in range(len(P.spectral.components)):
        #     axs[0].plot(P.Z[0, :], sol[i, 0, :].real, label=f'{P.index_to_name[i]}')
        #     axs[1].plot(P.X[:, 0], sol[i, :, 0].real, label=f'{P.index_to_name[i]}')
        # axs[0].plot(P.Z[0, :], expect['v'][0, :], '--')
        # axs[0].plot(P.Z[0, :], expect['T'][0, :], '--')
        idx = P.iT
        axs[0].plot(P.Z[0, :], sol[idx][0, :], label='want')
        axs[0].plot(P.Z[0, :], rhs[idx][0, :], label='have', ls='--')
        axs[1].plot(P.Z[0, :], rhs[P.ip][0, :] - sol[P.ip][0, :])
        axs[1].plot(P.Z[0, :], rhs[idx][0, :] - sol[idx][0, :], label='Tz')
        axs[0].legend()
        axs[1].legend()
        plt.show()

    for component in ['T', 'v']:
        i = P.index(component)
        poly = np.polynomial.Chebyshev(sol_hat[i][0])
        assert np.isclose(
            poly(-1), BCs[f'{component}_bottom']
        ), f'unexpected bottom bc in {component}: Deviation: {poly(-1).real- BCs[f"{component}_bottom"]:.2e}'
        assert np.isclose(
            poly(1), BCs[f'{component}_top']
        ), f'unexpected top bc in {component}: Deviation {poly(1).real-BCs[f"{component}_top"]:.2e}'
    assert np.isclose(pressure_integral.real, BCs['p_integral']), f'Got unexpected {pressure_integral.real=:.2e}!'
    for i in [P.iu]:
        assert np.allclose(sol[i], zero), f'Got non-zero values for {P.index_to_name[i]}'


@pytest.mark.base
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
@pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
def test_vorticity(nx, nz, cheby_mode, direction):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    assert nz > 3

    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode)

    u = P.u_init

    if direction == 'x':
        u[P.iv] = np.sin(P.X)
        u[P.iu] = np.cos(P.X)
        expect = np.cos(P.X)
    elif direction == 'z':
        u[P.iv] = P.Z**2
        u[P.iu] = P.Z**3
        expect = 3 * P.Z**2
    elif direction == 'mixed':
        u[P.iv] = np.sin(P.X) * P.Z**2
        u[P.iu] = np.cos(P.X) * P.Z**3
        expect = np.cos(P.X) * P.Z**2 + np.cos(P.X) * 3 * P.Z**2
    else:
        raise NotImplementedError

    assert np.allclose(P.compute_vorticity(u), expect)


# @pytest.mark.base
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


@pytest.mark.base
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
    # violations = P.compute_constraint_violation(u0)

    # for key in violations.keys():
    #     assert np.allclose(violations[key], 0), f'Violation of constraints in {key}!'

    # test BCs
    expect = {}
    expect[P.iT] = (BCs['T_top'] - BCs['T_bottom']) / 2 * P.Z + (BCs['T_top'] + BCs['T_bottom']) / 2.0
    expect[P.iv] = (BCs['v_top'] - BCs['v_bottom']) / 2 * P.Z + (BCs['v_top'] + BCs['v_bottom']) / 2.0
    for i in [P.iT, P.iv]:
        assert np.allclose(u0[i][:, 0], expect[i][:, 0]), f'Error in BCs in initial conditions of {P.index_to_name[i]}'
        assert np.allclose(
            u0[i][:, -1], expect[i][:, -1]
        ), f'Error in BCs in initial conditions of {P.index_to_name[i]}'


@pytest.mark.base
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

        plt.show()

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


@pytest.mark.base
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
        nx=2**1,
        nz=2**2,
        cheby_mode='T2U',
        comm=comm,
        solver_type='direct',
        left_preconditioner=True,
        right_preconditioning='D2T',
        Rayleigh=1.0,
    )

    u0 = P.u_exact(noise_level=1e-3, sigma=1.0)
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


@pytest.mark.base
@pytest.mark.parametrize('nx', [32])
@pytest.mark.parametrize('nz', [32])
@pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
def test_solver(nx, nz, cheby_mode, plotting=False):
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
        left_preconditioner=True,
        right_preconditioning='D2T',
    )  # , Rayleigh=4e3)

    def IMEX_Euler(_u, dt):
        f = P.eval_f(_u)
        un = P.solve_system(_u + dt * f.expl, dt)
        # un = P.solve_system(_u, dt)
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

    # P_heat = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode, Rayleigh=1e-8)
    # poisson = P_heat.solve_system(P_heat.u_exact(0, 1e1), 1e7)
    # expect_poisson = P_heat.u_exact(0, 0)
    # compute_errors(poisson, expect_poisson, 'Poisson')

    # u_static = P.u_exact(noise_level=0)
    # static = P.solve_system(u_static, 1e-0)
    # compute_errors(u_static, static, 'static configuration')

    u0 = P.u_exact(noise_level=1e-3, sigma=0.0)
    small_dt = P.solve_system(u0, 1e0)
    compute_errors(u0, small_dt, 'tiny step size', 1e-9)
    # P.plot(u0, quantity='p')
    # P.plot(small_dt, quantity='Tz')
    # plt.show()
    # return None

    dt = 1e-1
    u0 = P.u_exact(noise_level=1e-4, sigma=0.0)
    forward = P.solve_system(u0, dt)
    f = P.eval_f(forward)
    backward = forward - dt * (f.impl)
    compute_errors(u0, backward, 'backward without convection', 1e-8, components=['T', 'u', 'v'])
    # P.plot(forward, quantity='vz')
    # plt.show()
    # return None

    dt = 1e-1
    u0 = P.u_exact(noise_level=1e-4)
    forward = IMEX_Euler(u0, dt)
    f = P.eval_f(forward)
    f_before = P.eval_f(u0)
    backward = forward - dt * (f.impl + f_before.expl)
    compute_errors(u0, backward, 'backward', 1e-6, components=['T', 'u', 'v'])

    if plotting:
        u = P.u_exact(noise_level=1e-3, sigma=0)
        t = 0
        nsteps = 1000
        dt = 0.25
        fig = P.get_fig()
        # P.plot(u, t, fig=fig, quantity='u')
        # plt.show()
        # return None

        for i in range(nsteps):
            t += dt
            u = IMEX_Euler(u, dt)

            P.plot(u, t, fig=fig, quantity='T')
            plt.pause(1e-8)
        plt.show()


if __name__ == '__main__':
    # test_limit_case('Pr->inf', plotting=True)
    # test_derivatives(64, 64, 'z', 'T2U')
    test_eval_f(8, 8, 'T2T', 'x')
    # test_BCs(2**1, 2**7, 'T2U', 0, 0, 2, 0.001, True)
    # test_solver(2**7, 2**6, 'T2U', plotting=True)
    # test_solver_small_step_size(True)
    # test_vorticity(4, 4, 'T2T', 'x')
    # test_linear_operator(2**4, 2**4, 'T2U', 'x')
    # test_initial_conditions(4, 5, 0, 1, 1, 1)
