import pytest


@pytest.mark.base
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
@pytest.mark.parametrize('axes', [(0,), (1,), (0, 1)])
def test_integration_matrix2D(nx, nz, variant, axes):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    helper = SpectralHelper()
    helper.add_axis(base='fft', N=nx)
    helper.add_axis(base='cheby', N=nz, mode=variant)
    helper.setup_fft()

    Z, X = helper.get_grid()

    conv = helper.get_basis_change_mat()
    S = helper.get_integration_matrix(axes=axes)

    u = np.sin(X) * Z**2 + np.cos(X) * Z**3
    if axes == (0,):
        expect = -np.cos(X) * Z**2 + np.sin(X) * Z**3
    elif axes == (1,):
        expect = np.sin(X) * 1 / 3 * Z**3 + np.cos(X) * 1 / 4 * Z**4
    elif axes == (0, 1):
        expect = -np.cos(X) * 1 / 3 * Z**3 + np.sin(X) * 1 / 4 * Z**4

    u_hat = helper.transform(u, axes=(0, 1))
    S_u_hat = (conv @ S @ u_hat.flatten()).reshape(u_hat.shape)
    S_u = helper.itransform(S_u_hat, axes=(1, 0))

    assert np.allclose(S_u, expect, atol=1e-12)


@pytest.mark.base
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
def test_differentiation_matrix2D(nx, nz, variant, direction):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.helpers.spectral_helper import ChebychovHelper, FFTHelper

    cheby = ChebychovHelper(nz, mode=variant, transform_type='dct')
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
@pytest.mark.parametrize('nx', [3, 8])
@pytest.mark.parametrize('nz', [3, 8])
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
def test_transform(nx, nz, bz):
    import numpy as np
    import scipy
    from pySDC.helpers.spectral_helper import SpectralHelper

    bx = 'fft'

    helper = SpectralHelper()
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=bz, N=nz)
    helper.setup_fft()

    u = np.random.random((nx, nz))
    axes = (1, 0)

    expect_trf = u.copy()
    for i in axes:
        base = helper.axes[i]
        expect_trf = base.transform(expect_trf, axis=i)
        print(i, expect_trf)

    trf = helper.transform(u, axes=axes)
    itrf = helper.itransform(trf, axes=axes)

    assert np.allclose(itrf, u), 'Backward transform is unexpected'
    assert np.allclose(expect_trf, trf), 'Forward transform is unexpected'


if __name__ == '__main__':
    # test_transform(4, 3, 'cheby')
    test_integration_matrix2D(4, 3, 'T2U', (0, 1))
