import pytest

PARAMS = {
    'nx': 2**7,
    'nz': 2**7,
}

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

    derivatives = P._compute_derivatives(u)
    i = P.iu

    for i in [P.iux, P.iTx]:
        assert np.allclose(derivatives[i], expect_x), f'Got unexpected x-derivative in component {i}'
    for i in [P.ivz, P.iTz]:
        assert np.allclose(derivatives[i], expect_z), f'Got unexpected z-derivative in component {i}'


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
    Pr, Ra = P.Pr, P.Ra
    cos, sin = np.cos, np.sin

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
    u[P.iux] = y_x
    u[P.ivz] = y_z
    u[P.iTx] = y_x
    u[P.iTz] = y_z

    f = P.eval_f(u)

    for i in [P.iux, P.ivz, P.iTx, P.iTz]:
        assert np.allclose(f.impl[i] + f.expl[i], 0), f'Non-zero time derivative in algebraic component {i}'

    f_expect = P.f_init
    f_expect.expl[P.iT] = -y * (y_x + y_z)
    f_expect.impl[P.iT] = y_xx + y_zz
    f_expect.expl[P.iu] = -y * (y_z + y_x)
    f_expect.impl[P.iu] = -Pr * y_x + y_xx
    f_expect.expl[P.iv] = -y * (y_z + y_x)
    f_expect.impl[P.iv] = -Pr * y_z + Pr * Ra * y + y_zz

    for i in range(u.shape[0]):
        assert np.allclose(f.impl[i], f_expect.impl[i]), f'Unexpected implicit function evaluation in component {i}'
        assert np.allclose(f.expl[i], f_expect.expl[i]), f'Unexpected explicit function evaluation in component {i}'


@pytest.mark.base
@pytest.mark.parametrize('nx', [1, 4])
@pytest.mark.parametrize('nz', [2])
@pytest.mark.parametrize('cheby_mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('T_top', [2])
@pytest.mark.parametrize('T_bottom', [3.14])
@pytest.mark.parametrize('v_top', [2.77])
@pytest.mark.parametrize('v_bottom', [-90])
def test_BCs(nx, nz, cheby_mode, T_top, T_bottom, v_top, v_bottom):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'T_top': T_top,
        'T_bottom': T_bottom,
        'v_top': v_top,
        'v_bottom': v_bottom,
        'p_top': 0,
        # 'u_top': 1,
        # 'u_bottom': -1,
    }
    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode, BCs=BCs)

    rhs = P._put_BCs_in_rhs(P.u_init)
    _A = 1e-1 * P.L + P.M
    A = P._put_BCs_in_matrix(_A)

    rhs_hat = P.transform(rhs).flatten()

    sol_hat = sp.linalg.spsolve(A, rhs_hat)
    sol = P.itransform(sol_hat.reshape(P.u_init.shape))

    expect = {}
    for q in ['T', 'v']:
        if nz == 2 or True:
            expect[q] = (BCs[f'{q}_top'] - BCs[f'{q}_bottom']) / 2 * P.Z + (BCs[f'{q}_top'] + BCs[f'{q}_bottom']) / 2
        else:
            raise NotImplementedError
    zero = np.zeros_like(expect['T'])

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(abs(_A.toarray()))
    # axs[1].imshow(abs(A.toarray()))
    for i in range(8):
        axs[0].plot(P.Z[0, :], sol[i, 0, :], label=f'{P.index_to_name[i]}')
        axs[1].plot(P.X[:, 0], sol[i, :, 0], label=f'{P.index_to_name[i]}')
    axs[0].plot(P.Z[0, :], expect['v'][0, :], '--')
    axs[0].plot(P.Z[0, :], expect['T'][0, :], '--')
    axs[0].legend()
    plt.show()

    for i in [P.iTx, P.iu, P.iux, P.ip]:
        assert np.allclose(sol[i], zero), f'Got non-zero values for {P.index_to_name[i]}'
    for i in [P.iT, P.iv]:
        assert np.allclose(sol[i], expect[P.index_to_name[i]]), f'Unexpected BCs in {P.index_to_name[i]}'


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

    assert np.allclose(P.compute_vorticiy(u), expect)


def test_linear_operator(nx, nz, cheby_mode, direction):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
    import matplotlib.pyplot as plt

    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode)

    u = P.u_init
    expect = P.u_init

    for i in [P.iu, P.iv, P.iT, P.ip]:
        if direction == 'x':
            u[i] = np.sin(P.X * (i + 1))
        else:
            raise NotImplementedError

    derivatives = P._compute_derivatives(u)
    for i in [P.iux, P.iTx, P.iTz, P.ivz]:
        u[i] = derivatives[i]

    if direction == 'x':
        expect[P.iu] = P.Pr * (-(P.ip + 1) * np.cos(P.X * (P.ip + 1)) - (P.iu + 1) ** 2 * np.sin(P.X * (P.iu + 1)))
        expect[P.iv] = P.Pr * P.Ra * u[P.iT]
        expect[P.iux] = 0.0
        expect[P.ivz] = -(P.iu + 1) * np.cos(P.X * (P.iu + 1))
        expect[P.iT] = -((P.iT + 1) ** 2) * np.sin(P.X * (P.iT + 1))
        expect[P.iTx] = 0
        expect[P.iTz] = 0
        expect[P.ip] = np.sin(P.X * (P.ip + 1)) * P.Z**2 / 2.0
    else:
        raise NotImplementedError

    u_hat = P.transform(u)
    Lu_hat = (P.L @ u_hat.flatten()).reshape(u.shape)
    Lu = P.itransform(Lu_hat)

    fig, axs = plt.subplots(1, 3)
    i = P.ip
    axs[0].pcolormesh(P.X, P.Z, u[i].real)
    axs[1].pcolormesh(P.X, P.Z, Lu[i].real)
    axs[2].pcolormesh(P.X, P.Z, expect[i].real)
    plt.show()

    for i in range(u.shape[0]):
        assert np.allclose(Lu[i], expect[i]), f'Got unexpected result in component {P.index_to_name[i]}'


def test_solver(nx, nz, cheby_mode):
    import numpy as np
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
    import matplotlib.pyplot as plt

    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode)

    zero = P.u_init
    u = P.u_exact()
    t = 0

    def IMEX_Euler(_u, dt):
        f = P.eval_f(_u)
        un = P.solve_system(_u + dt * f.expl, dt)
        return un

    small_dt = P.solve_system(zero, 1e-9)
    for i in range(small_dt.shape[0]):
        error = abs(zero[i] - small_dt[i])
        print(error)
        # assert np.allclose(error, 0, atol=1e-7), f'{error=:.2e} in {P.index_to_name[i]} when solving with small step size'

    nsteps = 1000
    dt = 2e0
    fig = P.get_fig()

    for i in range(nsteps):
        t += dt
        u = IMEX_Euler(u, dt)

        P.plot(u, t, fig=fig)
        # im = plt.pcolormesh(P.X, P.Z, P.compute_vorticiy(u).real)
        # im = plt.pcolormesh(P.X, P.Z, u[P.iT].real)
        # plt.title(t)
        # plt.colorbar(im)
        plt.pause(1e-8)
    plt.show()


if __name__ == '__main__':
    # test_derivatives(4, 4, 'mixed', 'T2U')
    # test_eval_f(128, 129, 'T2T', 'z')
    # test_BCs(2**6, 2**6, 'T2T', 1, -2, 3, 2)
    # test_solver(2**6, 2**6, 'T2T')
    # test_vorticity(4, 4, 'T2T', 'x')
    test_linear_operator(2**7, 2**7, 'T2T', 'x')
