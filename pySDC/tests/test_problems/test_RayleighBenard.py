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


def test_BCs(nx, nz, cheby_mode):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    BCs = {
        'T_top': -1,
    }
    P = RayleighBenard(nx=nx, nz=nz, cheby_mode=cheby_mode, BCs=BCs)

    rhs = P._put_BCs_in_rhs(P.u_init).flatten()
    _A = P.L + P.M
    A = P._put_BCs_in_matrix(_A)
    print(A.toarray())
    # print(P.L.toarray() + P.M.toarray())
    print(rhs)

    sol_hat = sp.linalg.spsolve(A, rhs)
    sol = P.itransform(sol_hat.reshape(P.u_init.shape))
    print(sol)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(abs(_A.toarray()))
    # axs[1].imshow(abs(A.toarray()))
    for i in range(8):
        axs[0].plot(P.Z[0, :], sol[i, 0, :], label='i')
    plt.show()


if __name__ == '__main__':
    # test_derivatives(4, 4, 'mixed', 'T2U')
    # test_eval_f(128, 129, 'T2T', 'z')
    test_BCs(1, 2**4, 'T2T')
