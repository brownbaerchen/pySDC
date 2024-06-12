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

    t = eval_poly(chebyt, coeffs, x)
    u_from_t = eval_poly(chebyu, T2U @ coeffs, x)

    assert np.allclose(u, t_from_u)
    assert np.allclose(t, u_from_t)


@pytest.mark.base
@pytest.mark.parametrize('name', ['T2U', 'T2D', 'U2D'])
def test_conversion_inverses(name):
    from pySDC.helpers.problem_helper import ChebychovHelper
    import numpy as np

    N = 8
    cheby = ChebychovHelper(N)
    P = cheby.get_conv(name)
    Pinv = cheby.get_conv(name[::-1])
    assert np.allclose((P @ Pinv).toarray(), np.diag(np.ones(N)))


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
def test_differentiation_matrix(N, variant):
    import numpy as np
    import scipy
    from pySDC.helpers.problem_helper import ChebychovHelper

    cheby = ChebychovHelper(N)
    x = np.cos(np.pi / N * (np.arange(N) + 0.5))
    coeffs = np.random.random(N)
    norm = cheby.get_norm()

    if variant == 'T2U':
        _D = cheby.get_T2U_differentiation_matrix()
        U2T = cheby.get_conv('U2T')

        D = U2T @ _D
    elif variant == 'T2T':
        D = cheby.get_T2T_differentiation_matrix(1)

    du = scipy.fft.idct(D @ coeffs / norm)
    exact = np.polynomial.Chebyshev(coeffs).deriv(1)(x)

    assert np.allclose(exact, du)


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


if __name__ == '__main__':
    test_conversion_inverses('T2D')
