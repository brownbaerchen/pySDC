import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_D2T_conversion_matrices(N):
    import numpy as np
    from pySDC.helpers.problem_helper import ChebychovHelper

    cheby = ChebychovHelper(N)

    x = np.linspace(-1, 1, N)
    D2T = cheby.get_conv('D2T')

    for i in range(N):
        coeffs = np.zeros(N)
        coeffs[i] = 1.0
        T_coeffs = D2T @ coeffs

        Dn = np.polynomial.Chebyshev(T_coeffs)(x)

        expect_left = (-1) ** i if i < 2 else 0
        expect_right = 1 if i < 2 else 0

        assert Dn[0] == expect_left
        assert Dn[-1] == expect_right


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_T_U_conversion(N):
    import numpy as np
    from scipy.special import chebyt, chebyu
    from pySDC.helpers.problem_helper import ChebychovHelper

    cheby = ChebychovHelper(N)

    T2U = cheby.get_conv('T2U')
    U2T = cheby.get_conv('U2T')

    coeffs = np.random.random(N)
    x = cheby.get_1dgrid()

    def eval_poly(poly, coeffs, x):
        return np.array([coeffs[i] * poly(i)(x) for i in range(len(coeffs))]).sum(axis=0)

    u = eval_poly(chebyu, coeffs, x)
    t_from_u = eval_poly(chebyt, U2T @ coeffs, x)
    t_from_u_r = eval_poly(chebyt, coeffs @ U2T.T, x)

    t = eval_poly(chebyt, coeffs, x)
    u_from_t = eval_poly(chebyu, T2U @ coeffs, x)
    u_from_t_r = eval_poly(chebyu, coeffs @ T2U.T, x)

    assert np.allclose(u, t_from_u)
    assert np.allclose(u, t_from_u_r)
    assert np.allclose(t, u_from_t)
    assert np.allclose(t, u_from_t_r)


@pytest.mark.base
@pytest.mark.parametrize('name', ['T2U', 'T2D', 'U2D', 'T2T'])
def test_conversion_inverses(name):
    from pySDC.helpers.problem_helper import ChebychovHelper
    import numpy as np

    N = 8
    cheby = ChebychovHelper(N)
    P = cheby.get_conv(name)
    Pinv = cheby.get_conv(name[::-1])
    assert np.allclose((P @ Pinv).toarray(), np.diag(np.ones(N)))


@pytest.mark.base
@pytest.mark.parametrize('N', [4])
@pytest.mark.parametrize('convs', [['D2T', 'T2U'], ['U2D', 'D2T'], ['T2U', 'U2D'], ['T2U', 'U2D', 'D2T']])
def test_multi_conversion(N, convs):
    from pySDC.helpers.problem_helper import ChebychovHelper
    import numpy as np

    N = 8
    cheby = ChebychovHelper(N)

    full_conv = cheby.get_conv(f'{convs[0][0]}2{convs[-1][-1]}')
    P = cheby.get_conv(convs[-1])
    for i in range(1, len(convs)):
        P = cheby.get_conv(convs[-i - 1]) @ P

    assert np.allclose(P.toarray(), full_conv.toarray())


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('variant', ['T2U', 'T2T', 'D2U'])
def test_differentiation_matrix(N, variant):
    import numpy as np
    import scipy
    from pySDC.helpers.problem_helper import ChebychovHelper

    cheby = ChebychovHelper(N)
    x = np.cos(np.pi / N * (np.arange(N) + 0.5))
    coeffs = np.random.random(N)
    norm = cheby.get_norm()

    if variant == 'T2U':
        D = cheby.get_T2U_differentiation_matrix()
        P = cheby.get_conv('T2T')
        Q = cheby.get_conv('U2T')
    elif variant == 'T2T':
        D = cheby.get_T2T_differentiation_matrix(1)
        P = cheby.get_conv('T2T')
        Q = cheby.get_conv('T2T')
    elif variant == 'D2U':
        D = cheby.get_T2U_differentiation_matrix() @ cheby.get_conv('D2T')
        Q = cheby.get_conv('U2T')
        P = cheby.get_conv('T2D')
    else:
        raise NotImplementedError

    du = scipy.fft.idct(Q @ D @ P @ coeffs / norm)
    exact = np.polynomial.Chebyshev(coeffs).deriv(1)(x)

    assert np.allclose(exact, du)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
def test_integration_matrix(N, variant):
    import numpy as np
    from pySDC.helpers.problem_helper import ChebychovHelper

    cheby = ChebychovHelper(N, mode=variant)
    coeffs = np.random.random(N)
    coeffs[-1] = 0

    D = cheby.get_integration_matrix()

    if variant == 'T2U':
        P = cheby.get_conv('U2T')
    elif variant == 'T2T':
        P = cheby.get_conv('T2T')
    else:
        raise NotImplementedError

    du = P @ D @ coeffs
    exact = np.polynomial.Chebyshev(coeffs).integ(1)

    assert np.allclose(exact.coef[:-1], du)


@pytest.mark.base
@pytest.mark.parametrize('nz', [4, 32])
@pytest.mark.parametrize('nx', [4, 32])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
def test_integration_matrix2D2(nx, nz, variant, direction):
    import numpy as np
    import scipy
    from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper

    fft = FFTHelper(nx)
    cheby = ChebychovHelper(nz, mode=variant)
    x = fft.get_1dgrid()
    z = cheby.get_1dgrid()
    Z, X = np.meshgrid(z, x)

    coeffs_z = np.random.random(nz)
    coeffs_z[-1] = 0

    coeffs_x = np.random.random(nx // 2)

    CZ, CX = np.meshgrid(coeffs_z, coeffs_x)

    if direction == 'z':
        u_hat = CZ
    elif direction == 'x':
        u_hat = CX
    elif direction == 'mixed':
        u_hat = CZ + CX
    else:
        raise NotImplementedError

    S = cheby.get_integration_matrix()
    P = cheby.get_conv(variant)

    Su_hat = S @ P @ coeffs_z
    Su = cheby.itransform(Su_hat)
    exact = np.polynomial.Chebyshev(coeffs_z).integ(1)
    exact.coef[0] = 0  # fix integration constant to 0

    assert np.allclose(exact(Z), Su)


# @pytest.mark.base
# @pytest.mark.parametrize('nx', [16])
# @pytest.mark.parametrize('nz', [16])
# @pytest.mark.parametrize('variant', ['T2U', 'T2T'])
# @pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
# def test_integration_matrix2D(nx, nz, variant, direction):
#     import numpy as np
#     import scipy.sparse as sp
#     from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper
#
#     cheby = ChebychovHelper(nz, mode=variant)
#     fft = FFTHelper(nx)
#     x = fft.get_1dgrid()
#     z = cheby.get_1dgrid()
#     Z, X = np.meshgrid(z, x)
#     conv1D = cheby.get_conv(variant)
#     convInv1D = cheby.get_conv(variant[::-1])
#
#     Ix = fft.get_Id()
#     Iz = convInv1D
#     conv = sp.kron(Ix, conv1D)
#     Sx = fft.get_integration_matrix()
#     Sz = cheby.get_integration_matrix()
#     # Sz = cheby.get_U2T_integration_matrix()
#     # print(Sz.toarray())
#
#     u = np.sin(X) * Z**2 + np.cos(X) * Z**3
#     u =  Z**2
#     if direction == 'x':
#         S = sp.kron(Sx, Iz)
#         expect = -np.cos(X) * Z**2 + np.sin(X) * Z**3
#     elif direction == 'z':
#         S = sp.kron(Ix, Sz)
#         expect = np.sin(X) * 1/3*Z**3 + np.cos(X) * 1/4 * Z**4
#         print(S.toarray())
#         print(Sz.toarray())
#         print(conv1D.toarray())
#         expect =  Z**3 / 3.
#     elif direction == 'mixed':
#         S = sp.kron(Ix, Sz) @ sp.kron(Sx, sp.eye(nz))
#         S = sp.kron(Ix, Sz @ conv1D) @ sp.kron(Ix, sp.eye(nz))
#         # expect = -np.cos(X) * Z**3 / 3 + np.sin(X) / 4 * Z**4
#
#         expect = np.sin(X) * Z**2  # + Z**3 + np.cos(2 * X)
#
#     u_hat = fft.transform(cheby.transform(u, axis=-1), axis=-2)
#     print('u_hat', u_hat)
#     S_u_hat = (S  @ conv@ u_hat.flatten()).reshape(u_hat.shape)
#     # S_u_hat[:,0] = 1/4.
#     print(S_u_hat)
#     print(fft.transform(cheby.transform(expect, axis=-1), axis=-2))
#     S_u = fft.itransform(cheby.itransform(S_u_hat, axis=-1), axis=-2)
#
#     import matplotlib.pyplot as plt
#
#     fig, axs = plt.subplots(1, 3)
#     axs[0].pcolormesh(X, Z, (expect).real)
#     im = axs[1].pcolormesh(X, Z, (S_u).real)
#     im = axs[2].pcolormesh(X, Z, (S_u - expect).real)
#     fig.colorbar(im)
#     plt.show()
#     assert np.allclose(S_u, expect, atol=1e-8)


@pytest.mark.base
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
def test_differentiation_matrix2D(nx, nz, variant, direction):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper

    cheby = ChebychovHelper(nz, mode=variant)
    fft = FFTHelper(nx)
    x = fft.get_1dgrid()
    z = cheby.get_1dgrid()
    Z, X = np.meshgrid(z, x)
    conv1D = cheby.get_conv(variant[::-1])
    convInv1D = cheby.get_conv(variant)

    Ix = fft.get_Id()
    Iz = convInv1D
    conv = sp.kron(Ix, conv1D)
    Dx = fft.get_differentiation_matrix()
    Dz = cheby.get_differentiation_matrix()

    u = np.sin(X) * Z**2 + Z**3 + np.cos(2 * X)
    if direction == 'x':
        D = sp.kron(Dx, Iz)
        expect = np.cos(X) * Z**2 - 2 * np.sin(2 * X)
    elif direction == 'z':
        D = sp.kron(Ix, Dz)
        expect = np.sin(X) * Z * 2 + Z**2 * 3
    elif direction == 'mixed':
        D = sp.kron(Ix, Dz) @ sp.kron(Dx, sp.eye(nz))
        expect = np.cos(X) * 2 * Z

    u_hat = fft.transform(cheby.transform(u, axis=-1), axis=-2)
    D_u_hat = (conv @ D @ u_hat.flatten()).reshape(u_hat.shape)
    D_u = fft.itransform(cheby.itransform(D_u_hat, axis=-1), axis=-2)

    assert np.allclose(D_u, expect, atol=1e-9)


@pytest.mark.base
@pytest.mark.parametrize('N', [4])
@pytest.mark.parametrize('d', [1, 2, 3])
def test_transform(N, d):
    import scipy
    import numpy as np
    from pySDC.helpers.problem_helper import ChebychovHelper

    cheby = ChebychovHelper(N)
    u = np.random.random((d, N))
    norm = cheby.get_norm()
    x = cheby.get_1dgrid()

    itransform = cheby.itransform(u, axis=-1)

    assert np.allclose(scipy.fft.dct(u, axis=-1) * norm, cheby.transform(u, axis=-1))
    assert np.allclose(scipy.fft.idct(u / norm, axis=-1), itransform)
    assert np.allclose(u.shape, itransform.shape)
    for i in range(d):
        assert np.allclose(np.polynomial.Chebyshev(u[i])(x), itransform[i])


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_norm(N):
    from pySDC.helpers.problem_helper import ChebychovHelper
    import numpy as np
    import scipy

    cheby = ChebychovHelper(N)
    coeffs = np.random.random(N)
    x = cheby.get_1dgrid()
    norm = cheby.get_norm()

    u = np.polynomial.Chebyshev(coeffs)(x)
    u_dct = scipy.fft.idct(coeffs / norm)
    coeffs_dct = scipy.fft.dct(u) * norm

    assert np.allclose(u, u_dct)
    assert np.allclose(coeffs, coeffs_dct)


@pytest.mark.base
@pytest.mark.parametrize('bc', [-1, 0, 1])
@pytest.mark.parametrize('mode', ['T2T', 'T2U', 'D2U'])
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('bc_val', [-99, 3.1415])
def test_tau_method(mode, bc, N, bc_val):
    '''
    solve u_x - u + tau P = 0, u(bc) = bc_val

    We choose P = T_N or U_N. We replace the last row in the matrix with the boundary condition to get a
    unique solution for the given resolution.

    The test verifies that the solution satisfies the perturbed equation and the boundary condition.
    '''
    from pySDC.helpers.problem_helper import ChebychovHelper
    import numpy as np
    import scipy.sparse as sp

    cheby = ChebychovHelper(N)
    x = cheby.get_1dgrid()

    coef = np.append(np.zeros(N - 1), [1])
    rhs = np.append(np.zeros(N - 1), [bc_val])

    if mode == 'T2T':
        P = np.polynomial.Chebyshev(coef)
        D = cheby.get_T2T_differentiation_matrix()
        Id = np.diag(np.ones(N))

        A = D - Id
        A[-1, :] = cheby.get_Dirichlet_BC_row_T(bc)

        sol_hat = np.linalg.solve(A, rhs)

        print(A)
        print(rhs)
        print(sol_hat)

    elif mode == 'T2U':
        T2U = cheby.get_conv('T2U')
        U2T = cheby.get_conv('U2T')

        P = np.polynomial.Chebyshev(U2T @ coef)
        D = cheby.get_T2U_differentiation_matrix()
        Id = T2U

        A = D - Id
        A[-1, :] = cheby.get_Dirichlet_BC_row_T(bc)

        sol_hat = sp.linalg.spsolve(A, rhs)

    elif mode == 'D2U':
        if bc == 0:
            return None

        U2T = cheby.get_conv('U2T')
        T2U = cheby.get_conv('T2U')
        D2T = cheby.get_conv('D2T')

        P = np.polynomial.Chebyshev(U2T @ coef)
        D = cheby.get_T2U_differentiation_matrix() @ D2T
        Id = T2U @ D2T

        A = D - Id
        A[-1, :] = cheby.get_Dirichlet_BC_row_D(bc)

        sol_hat = D2T @ sp.linalg.spsolve(A, rhs)

    else:
        raise NotImplementedError

    sol_poly = np.polynomial.Chebyshev(sol_hat)
    d_sol_poly = sol_poly.deriv(1)
    x = np.linspace(-1, 1, 100)

    assert np.isclose(sol_poly(bc), bc_val), 'Solution does not satisfy boundary condition'

    tau = (d_sol_poly(x) - sol_poly(x)) / P(x)
    assert np.allclose(tau, tau[0]), 'Solution does not satisfy perturbed equation'


@pytest.mark.base
@pytest.mark.parametrize('bc', [-1, 1])
@pytest.mark.parametrize('mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
@pytest.mark.parametrize('bc_val', [-2, 1.0])
def test_tau_method2D(mode, bc, nz, nx, bc_val, plotting=False):
    '''
    solve u_z - 0.1u_xx -u_x + tau P = 0, u(bc) = sin(bc_val*x) -> space-time discretization of advection-diffusion
    problem. We do FFT in x-direction and Chebychov in z-direction.
    '''
    from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper
    import numpy as np
    import scipy.sparse as sp

    cheby = ChebychovHelper(nz, mode=mode)
    fft = FFTHelper(nx)

    # generate grid
    x = fft.get_1dgrid()
    z = cheby.get_1dgrid()
    Z, X = np.meshgrid(z, x)

    # put BCs in right hand side
    bcs = np.sin(bc_val * x)
    rhs = np.zeros_like(X)
    rhs[:, -1] = bcs
    rhs_hat = fft.transform(rhs, axis=-2)  # the rhs is already in Chebychov spectral space

    # generate matrices
    Dx = sp.linalg.matrix_power(fft.get_differentiation_matrix(), 2) * 1e-1 + fft.get_differentiation_matrix()
    Ix = fft.get_Id()
    Dz = cheby.get_differentiation_matrix()
    Iz = cheby.get_Id()
    A = sp.kron(Ix, Dz) - sp.kron(Dx, Iz)

    # put BCs in the system matrix
    BCz = sp.eye(nz, format='lil') * 0
    BCz[-1, :] = cheby.get_Dirichlet_BC_row_T(bc)
    BC = sp.kron(Ix, BCz, format='lil')
    A[BC != 0] = BC[BC != 0]

    # solve the system
    sol_hat = (sp.linalg.spsolve(A, rhs_hat.flatten())).reshape(rhs.shape)

    # transform back to real space
    _sol = fft.itransform(sol_hat, axis=-2).real
    sol = cheby.itransform(_sol, axis=-1)

    # construct polynomials for testing
    polys = [np.polynomial.Chebyshev(_sol[i, :]) for i in range(nx)]
    # d_polys = [me.deriv(1) for me in polys]
    # _z = np.linspace(-1, 1, 100)

    if plotting:
        import matplotlib.pyplot as plt

        im = plt.pcolormesh(X, Z, sol)
        plt.colorbar(im)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()

    for i in range(nx):

        assert np.isclose(polys[i](bc), bcs[i]), f'Solution does not satisfy boundary condition x={x[i]}'

        assert np.allclose(
            polys[i](z), sol[i, :]
        ), f'Solution is incorrectly transformed back to real space at x={x[i]}'

        # coef = np.append(np.zeros(nz - 1), [1])
        # Pz = np.polynomial.Chebyshev(coef)
        # tau = (d_polys[i](_z) - polys[i](_z)) / Pz(_z)
        # plt.plot(_z, tau)
        # plt.show()
        # assert np.allclose(tau, tau[0]), f'Solution does not satisfy perturbed equation at x={x[i]}'


@pytest.mark.base
@pytest.mark.parametrize('mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('bc_val', [4.0])
def test_tau_method2D_diffusion(mode, nz, nx, bc_val, plotting=False):
    '''
    Solve a Poisson problem with funny Dirichlet BCs in z-direction and periodic in x-direction.
    '''
    from pySDC.helpers.problem_helper import ChebychovHelper, FFTHelper
    import numpy as np
    import scipy.sparse as sp

    cheby = ChebychovHelper(nz, mode=mode)
    fft = FFTHelper(nx)

    # generate grid
    x = fft.get_1dgrid()
    z = cheby.get_1dgrid()
    Z, X = np.meshgrid(z, x)

    # put BCs in right hand side
    rhs = np.zeros((2, nx, nz))  # components u and u_x
    rhs[0, :, -1] = np.sin(bc_val * x) + 1
    rhs[1, :, -1] = 3 * np.exp(-((x - 3.6) ** 2)) + np.cos(x)
    rhs_hat = fft.transform(rhs, axis=-2)  # the rhs is already in Chebychov spectral space

    # generate 1D matrices
    Dx = fft.get_differentiation_matrix()
    Ix = fft.get_Id()
    Dz = cheby.get_differentiation_matrix()
    Iz = cheby.get_Id()

    # generate 2D matrices
    D = sp.kron(Ix, Dz) + sp.kron(Dx, Iz)
    I = sp.kron(Ix, Iz)
    O = I * 0

    # generate system matrix
    A = sp.bmat([[O, D], [D, -I]], format='lil')

    # generate BC matrices
    BCa = sp.eye(nz, format='lil') * 0
    BCa[-1, :] = cheby.get_Dirichlet_BC_row_T(-1)
    BCa = sp.kron(Ix, BCa, format='lil')

    BCb = sp.eye(nz, format='lil') * 0
    BCb[-1, :] = cheby.get_Dirichlet_BC_row_T(1)
    BCb = sp.kron(Ix, BCb, format='lil')
    BC = sp.bmat([[BCa, O], [BCb, O]], format='lil')

    # put BCs in the system matrix
    A[BC != 0] = BC[BC != 0]

    # solve the system
    sol_hat = (sp.linalg.spsolve(A, rhs_hat.flatten())).reshape(rhs.shape)

    # transform back to real space
    _sol = fft.itransform(sol_hat, axis=-2).real
    sol = cheby.itransform(_sol, axis=-1)

    polys = [np.polynomial.Chebyshev(_sol[0, i, :]) for i in range(nx)]

    if plotting:
        import matplotlib.pyplot as plt

        im = plt.pcolormesh(X, Z, sol[0])
        plt.colorbar(im)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()

    for i in range(nx):

        assert np.isclose(polys[i](-1), rhs[0, i, -1]), f'Solution does not satisfy lower boundary condition x={x[i]}'
        assert np.isclose(polys[i](1), rhs[1, i, -1]), f'Solution does not satisfy upper boundary condition x={x[i]}'

        assert np.allclose(
            polys[i](z), sol[0, i, :]
        ), f'Solution is incorrectly transformed back to real space at x={x[i]}'


if __name__ == '__main__':
    # test_tau_method('T2T', -1.0, N=5, bc_val=3.0)
    # test_tau_method2D('T2T', -1, nx=2**7, nz=2**6, bc_val=4.0, plotting=True)
    test_integration_matrix(5, 'T2U')
    # test_integration_matrix2D(2**0, 2**2, 'T2U', 'z')
    # test_differentiation_matrix2D(2**7, 2**7, 'T2U', 'mixed')
