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
@pytest.mark.parametrize('axes', [(0,), (1,), (0, 1)])
def test_differentiation_matrix2D(nx, nz, variant, axes):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    helper = SpectralHelper()
    helper.add_axis(base='fft', N=nx)
    helper.add_axis(base='cheby', N=nz, mode=variant)
    helper.setup_fft()

    Z, X = helper.get_grid()
    conv = helper.get_basis_change_mat()
    D = helper.get_differentiation_matrix(axes)

    u = np.sin(X) * Z**2 + Z**3 + np.cos(2 * X)
    if axes == (0,):
        expect = np.cos(X) * Z**2 - 2 * np.sin(2 * X)
    elif axes == (1,):
        expect = np.sin(X) * Z * 2 + Z**2 * 3
    elif axes == (0, 1):
        expect = np.cos(X) * 2 * Z

    u_hat = helper.transform(u, axes=(0, 1))
    D_u_hat = (conv @ D @ u_hat.flatten()).reshape(u_hat.shape)
    D_u = helper.itransform(D_u_hat, axes=(1, 0))

    assert np.allclose(D_u, expect, atol=1e-12)


@pytest.mark.base
@pytest.mark.parametrize('nx', [3, 8])
@pytest.mark.parametrize('nz', [3, 8])
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
def test_transform(nx, nz, bz, useMPI=False, **kwargs):
    import numpy as np
    import scipy
    from pySDC.helpers.spectral_helper import SpectralHelper

    bx = 'fft'

    helper = SpectralHelper()
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=bz, N=nz)
    helper.setup_fft(useMPI=useMPI)

    u = np.random.random((nx, nz))
    axes = (1, 0)

    expect_trf = u.copy()
    for i in axes:
        base = helper.axes[i]
        norm = base.N if useMPI else 1.0
        expect_trf = base.transform(expect_trf, axis=i) / norm

    trf = helper.transform(u, axes=axes)
    itrf = helper.itransform(trf, axes=axes)

    assert np.allclose(itrf, u), 'Backward transform is unexpected'
    assert np.allclose(expect_trf, trf), 'Forward transform is unexpected'


def run_MPI_test(num_procs, **kwargs):
    import os
    import subprocess

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_procs} python {__file__}"

    for key, value in kwargs.items():
        cmd += f' --{key}={value}'
    p = subprocess.Popen(cmd.split(), env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [3, 8])
@pytest.mark.parametrize('nz', [3, 8])
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
def test_transform_MPI(nx, nz, bz):
    run_MPI_test(num_procs=1, test='transform', nx=nx, nz=nz, bz=bz)


if __name__ == '__main__':
    str_to_bool = lambda me: False if me == 'False' else True

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, help='Dof in x direction')
    parser.add_argument('--nz', type=int, help='Dof in z direction')
    parser.add_argument('--bz', type=str, help='Base in z direction')
    parser.add_argument('--bx', type=str, help='Base in x direction')
    parser.add_argument('--test', type=str, help='type of test', choices=['transform'])
    parser.add_argument('--useMPI', type=str_to_bool, help='use MPI or not', choices=[True, False], default=True)
    args = parser.parse_args()

    if args.test == 'transform':
        test_transform(**vars(args))

    # test_transform_MPI(4, 3, 'cheby')
    # test_differentiation_matrix2D(2, 2, 'T2U', (0,))
