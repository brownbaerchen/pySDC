import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [8, 64])
def test_differentiation_matrix(N, plot=False):
    import numpy as np
    from pySDC.helpers.spectral_helper import FFTHelper

    helper = FFTHelper(N=N)

    x = helper.get_1dgrid()
    D = helper.get_differentiation_matrix()

    u = np.zeros_like(x)
    expect = np.zeros_like(u)

    num_coef = N // 2
    coeffs = np.random.random((2, N))
    for i in range(num_coef):
        u += coeffs[0, i] * np.sin(i * x)
        u += coeffs[1, i] * np.cos(i * x)
        expect += coeffs[0, i] * i * np.cos(i * x)
        expect -= coeffs[1, i] * i * np.sin(i * x)

    u_hat = np.fft.fft(u)
    Du_hat = D @ u_hat
    Du = np.fft.ifft(Du_hat)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(x, u)
        plt.plot(x, Du)
        plt.plot(x, expect, '--')
        plt.show()

    assert np.allclose(expect, Du)


@pytest.mark.base
def test_transform(N=8):
    import numpy as np
    from pySDC.helpers.spectral_helper import FFTHelper

    u = np.random.random(N)
    helper = FFTHelper(N=N)
    u_hat = helper.transform(u)
    assert np.allclose(u, helper.itransform(u_hat))


@pytest.mark.base
@pytest.mark.parametrize('N', [8, 64])
def test_integration_matrix(N, plot=False):
    import numpy as np
    from pySDC.helpers.spectral_helper import FFTHelper

    helper = FFTHelper(N=N)

    x = helper.get_1dgrid()
    D = helper.get_integration_matrix()

    u = np.zeros_like(x)
    expect = np.zeros_like(u)

    num_coef = N // 2 - 1
    coeffs = np.random.random((2, N))
    for i in range(1, num_coef + 1):
        u += coeffs[0, i] * np.sin(i * x)
        u += coeffs[1, i] * np.cos(i * x)
        expect -= coeffs[0, i] / i * np.cos(i * x)
        expect += coeffs[1, i] / i * np.sin(i * x)

    u_hat = np.fft.fft(u)
    Du_hat = D @ u_hat
    Du = np.fft.ifft(Du_hat)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(x, u)
        plt.plot(x, Du)
        plt.plot(x, expect, '--')
        plt.show()

    assert np.allclose(expect, Du)


if __name__ == '__main__':
    test_integration_matrix(8, True)
