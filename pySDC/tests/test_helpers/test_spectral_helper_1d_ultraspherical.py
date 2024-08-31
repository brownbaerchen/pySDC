import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 7, 32])
@pytest.mark.parametrize('p', [1, 2, 3, 4])
def test_differentiation_matrix(N, p):
    import numpy as np
    import scipy
    from pySDC.helpers.spectral_helper import Ultraspherical

    helper = Ultraspherical(N)
    x = helper.get_1dgrid()
    coeffs = np.random.random(N)
    norm = helper.get_norm()

    D = helper.get_differentiation_matrix(p=p)
    Q = helper.get_conv(p_out=0, p_in=p)
    Q = helper.get_basis_change_matrix(p)

    du = scipy.fft.idct(Q @ D @ coeffs / norm)
    exact = np.polynomial.Chebyshev(coeffs).deriv(p)(x)
    P = helper.get_conv(p_out=p)

    assert np.allclose(exact, du)


if __name__ == '__main__':
    test_differentiation_matrix(6, 2)
