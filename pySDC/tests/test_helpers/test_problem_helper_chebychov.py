import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_D2T_conversion_matrices(N):
    import numpy as np
    from pySDC.helpers.problem_helper import ChebychovHelper

    Cheby = ChebychovHelper(N)

    x = np.linspace(-1, 1, N)
    T2D, D2T = Cheby.get_D2T_conversion_matrices()

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

    Cheby = ChebychovHelper(N)

    T2U, U2T = Cheby.getT2U_converstion_matrices()

    coeffs = np.random.random(N)
    x = Cheby.get_1dgrid()

    def eval_poly(poly, coeffs, x):
        return np.array([coeffs[i] * poly(i)(x) for i in range(len(coeffs))]).sum(axis=0)

    u = eval_poly(chebyu, coeffs, x)
    t_from_u = eval_poly(chebyt, U2T @ coeffs, x)

    t = eval_poly(chebyt, coeffs, x)
    u_from_t = eval_poly(chebyu, T2U @ coeffs, x)

    assert np.allclose(u, t_from_u)
    assert np.allclose(t, u_from_t)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
def test_differentiation_matrix(N):
    import numpy as np
    from pySDC.helpers.problem_helper import ChebychovHelper

    Cheby = ChebychovHelper(N)

    D = Cheby.get_differentiation_matrix()
    T2U, U2T = Cheby.getT2U_converstion_matrices()
    x = np.cos(np.pi / N * (np.arange(N) + 0.5))

    coeffs = np.random.random(N)

    du = np.polynomial.Chebyshev(U2T @ D @ coeffs)(x)
    exact = np.polynomial.Chebyshev(coeffs).deriv(1)(x)

    assert np.allclose(exact, du)


if __name__ == '__main__':
    test_differentiation_matrix(4)
