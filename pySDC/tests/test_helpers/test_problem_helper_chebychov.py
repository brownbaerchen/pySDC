import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_D2T_conversion_matrices(N):
    import numpy as np
    from pySDC.helpers.problem_helper import ChebychovHelper

    cheby = ChebychovHelper(N, 1)

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

    cheby = ChebychovHelper(N, 1)

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
    cheby = ChebychovHelper(N, 1)
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
    cheby = ChebychovHelper(N, 1)

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

    cheby = ChebychovHelper(N, 1)
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
@pytest.mark.parametrize('N', [4])
def test_dct(N):
    import scipy
    import numpy as np
    from pySDC.helpers.problem_helper import ChebychovHelper

    cheby = ChebychovHelper(N)
    u = np.random.random(N)
    norm = cheby.get_norm()

    assert np.allclose(scipy.fft.dct(u) * norm, cheby.dct(u))
    assert np.allclose(scipy.fft.idct(u / norm), cheby.idct(u))


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_norm(N):
    from pySDC.helpers.problem_helper import ChebychovHelper
    import numpy as np
    import scipy

    cheby = ChebychovHelper(N, 1)
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
@pytest.mark.parametrize('method', ['T2T', 'T2U', 'D2U'])
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('bc_val', [-99, 3.1415])
def test_tau_method(method, bc, N, bc_val):
    '''
    solve u_x - u + tau P = 0, u(bc) = bc_val

    We choose P = T_N or U_N. We replace the last row in the matrix with the boundary condition to get a
    unique solution for the given resolution.

    The test verifies that the solution satisfies the perturbed equation and the boundary condition.
    '''
    from pySDC.helpers.problem_helper import ChebychovHelper
    import numpy as np
    import scipy.sparse as sp

    cheby = ChebychovHelper(N, 1)
    x = cheby.get_1dgrid()

    coef = np.append(np.zeros(N - 1), [1])
    rhs = np.append(np.zeros(N - 1), [bc_val])

    if method == 'T2T':
        P = np.polynomial.Chebyshev(coef)
        D = cheby.get_T2T_differentiation_matrix()
        Id = np.diag(np.ones(N))

        A = D - Id
        A[-1, :] = cheby.get_Dirichlet_BC_row_T(bc)

        sol_hat = np.linalg.solve(A, rhs)

    elif method == 'T2U':
        T2U = cheby.get_conv('T2U')
        U2T = cheby.get_conv('U2T')

        P = np.polynomial.Chebyshev(U2T @ coef)
        D = cheby.get_T2U_differentiation_matrix()
        Id = T2U

        A = D - Id
        A[-1, :] = cheby.get_Dirichlet_BC_row_T(bc)

        sol_hat = sp.linalg.spsolve(A, rhs)

    elif method == 'D2U':
        if bc == 0:
            return None

        U2T = cheby.get_conv('U2T')
        T2U = cheby.get_conv('T2U')
        D2T = cheby.get_conv('D2T')

        P = np.polynomial.Chebyshev(U2T @ coef)
        D = cheby.get_T2U_differentiation_matrix()
        Id = T2U

        A = (D - Id) @ D2T
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


if __name__ == '__main__':
    test_tau_method('D2U', -1.0, N=4, bc_val=3.0)
