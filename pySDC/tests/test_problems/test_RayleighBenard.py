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
        assert np.allclose(derivatives[i].real, expect_x), f'Got unexpected x-derivative in component {i}'
    for i in [P.ivz, P.iTz]:
        assert np.allclose(derivatives[i].real, expect_z.real), f'Got unexpected z-derivative in component {i}'


if __name__ == '__main__':
    test_derivatives(4, 4, 'mixed', 'T2U')
