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

    D = helper.get_differentiation_matrix(p=p)
    Q = helper.get_basis_change_matrix(p)

    du = helper.itransform(Q @ D @ coeffs)
    exact = np.polynomial.Chebyshev(coeffs).deriv(p)(x)
    P = helper.get_conv(p_out=p)

    assert np.allclose(exact, du)


@pytest.mark.base
@pytest.mark.parametrize('N', [6, 33])
@pytest.mark.parametrize('deg', [1, 3])
def test_poisson_problem(N, deg):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.helpers.spectral_helper import Ultraspherical

    a = 0
    b = 4

    helper = Ultraspherical(N)
    x = helper.get_1dgrid()

    f = x**deg * (deg + 1) * (deg + 2) * (a - b) / 2

    Dxx = helper.get_differentiation_matrix(p=2)
    BC_l = helper.get_Dirichlet_BC_row_T(x=-1)
    BC_r = helper.get_Dirichlet_BC_row_T(x=1)
    P = helper.get_Id()

    A = Dxx.tolil()
    A[-1, :] = BC_l
    A[-2, :] = BC_r
    A = A.tocsr()

    rhs = P @ helper.transform(f)
    rhs[-2] = a
    rhs[-1] = b

    u_hat = sp.linalg.spsolve(A, rhs)

    u = helper.itransform(u_hat)

    u_exact = (a - b) / 2 * x ** (deg + 2) + (b + a) / 2

    assert np.allclose(u_hat[deg + 3 :], 0)
    assert np.allclose(u_exact, u)


if __name__ == '__main__':
    # test_differentiation_matrix(6, 2)
    test_poisson_problem(6, 1)
