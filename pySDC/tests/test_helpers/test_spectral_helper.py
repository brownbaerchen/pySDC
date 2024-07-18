import pytest


@pytest.mark.base
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
@pytest.mark.parametrize('axes', [(-2,), (-1,), (-2, -1)])
def test_integration_matrix2D(nx, nz, variant, axes):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    helper = SpectralHelper()
    helper.add_axis(base='fft', N=nx)
    helper.add_axis(base='cheby', N=nz, mode=variant)
    helper.setup_fft()

    Z, X = helper.get_grid()

    conv = helper.get_basis_change_matrix()
    S = helper.get_integration_matrix(axes=axes)

    u = helper.u_init
    u[0, ...] = np.sin(X) * Z**2 + np.cos(X) * Z**3
    if axes == (-2,):
        expect = -np.cos(X) * Z**2 + np.sin(X) * Z**3
    elif axes == (-1,):
        expect = np.sin(X) * 1 / 3 * Z**3 + np.cos(X) * 1 / 4 * Z**4
    elif axes == (-2, -1):
        expect = -np.cos(X) * 1 / 3 * Z**3 + np.sin(X) * 1 / 4 * Z**4

    u_hat = helper.transform(u, axes=(-2, -1))
    S_u_hat = (conv @ S @ u_hat.flatten()).reshape(u_hat.shape)
    S_u = helper.itransform(S_u_hat, axes=(-1, -2))

    assert np.allclose(S_u, expect, atol=1e-12)


@pytest.mark.base
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
@pytest.mark.parametrize('axes', [(-2,), (-1,), (-2, -1)])
def test_differentiation_matrix2D(nx, nz, variant, axes):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    helper = SpectralHelper()
    helper.add_axis(base='fft', N=nx)
    helper.add_axis(base='cheby', N=nz, mode=variant)
    helper.setup_fft()

    Z, X = helper.get_grid()
    conv = helper.get_basis_change_matrix()
    D = helper.get_differentiation_matrix(axes)

    u = helper.u_init
    u[0, ...] = np.sin(X) * Z**2 + Z**3 + np.cos(2 * X)
    if axes == (-2,):
        expect = np.cos(X) * Z**2 - 2 * np.sin(2 * X)
    elif axes == (-1,):
        expect = np.sin(X) * Z * 2 + Z**2 * 3
    elif axes == (-2, -1):
        expect = np.cos(X) * 2 * Z
    else:
        raise NotImplementedError

    u_hat = helper.transform(u, axes=(-2, -1))
    D_u_hat = (conv @ D @ u_hat.flatten()).reshape(u_hat.shape)
    D_u = helper.itransform(D_u_hat, axes=(-1, -2))

    assert np.allclose(D_u, expect, atol=1e-12)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('base', ['cheby'])
@pytest.mark.parametrize('type', ['diff', 'int'])
def test_matrix1D(N, base, type):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    coeffs = np.random.random(N)

    helper = SpectralHelper()
    helper.add_axis(base=base, N=N)
    helper.setup_fft()

    x = helper.get_grid()

    if type == 'diff':
        D = helper.get_differentiation_matrix(axes=(-1,))
    elif type == 'int':
        D = helper.get_integration_matrix(axes=(-1,))

    C = helper.get_basis_change_matrix()

    u = helper.u_init
    u[0] = C @ D @ coeffs
    du = helper.itransform(u, axes=(-1,))

    if type == 'diff':
        exact = np.polynomial.Chebyshev(coeffs).deriv(1)(x)
    elif type == 'int':
        exact = np.polynomial.Chebyshev(coeffs).integ(1)(x)

    assert np.allclose(exact, du)


@pytest.mark.base
@pytest.mark.parametrize('nx', [3, 8])
@pytest.mark.parametrize('nz', [3, 8])
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
@pytest.mark.parametrize('axes', [(-1,), (-1, -2)])
def test_transform(nx, nz, bz, axes, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    bx = 'fft'

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.rank
    else:
        comm = None

    helper = SpectralHelper(comm=comm)
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=bz, N=nz)
    helper.setup_fft(useMPI=useMPI)

    u = helper.u_init
    u[...] = np.random.random(u.shape)

    u_all = np.empty(shape=(1, nx, nz), dtype=u.dtype)

    if useMPI:
        rank = comm.rank
        u_all[...] = (
            np.array(comm.allgather(u.transpose(0, 2, 1)))
            .reshape(np.array(helper.global_shape)[[0, 2, 1]])
            .transpose(0, 2, 1)
        )
        if comm.size == 1:
            assert np.allclose(u_all, u)
    else:
        rank = 0
        u_all[...] = u

    expect_trf = u_all.copy()
    for i in axes:
        base = helper.axes[i]
        norm = base.N if useMPI else 1.0
        expect_trf = base.transform(expect_trf, axis=i) / norm

    trf = helper.transform(u, axes=axes)
    # itrf = helper.itransform(trf, axes=axes)

    expect_local = expect_trf[:, :, trf.shape[2] * rank : trf.shape[2] * (rank + 1)]

    # print(u / itrf)
    print(expect_local)
    print(trf)
    assert np.allclose(expect_local, trf), 'Forward transform is unexpected'
    # assert np.allclose(itrf, u), 'Backward transform is unexpected'


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
@pytest.mark.parametrize('nx', [2, 8])
@pytest.mark.parametrize('nz', [2, 8])
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
@pytest.mark.parametrize('num_procs', [1, 2])
@pytest.mark.parametrize('axes', ["-1", "-1,-2"])
def test_transform_MPI(nx, nz, bz, num_procs, axes):
    run_MPI_test(num_procs=num_procs, test='transform', nx=nx, nz=nz, bz=bz, axes=axes)


@pytest.mark.base
@pytest.mark.parametrize('bc', [-1, 0, 1])
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('bc_val', [-99, 3.1415])
def test_tau_method(bc, N, bc_val):
    '''
    solve u_x - u + tau P = 0, u(bc) = bc_val

    We choose P = T_N or U_N. We replace the last row in the matrix with the boundary condition to get a
    unique solution for the given resolution.

    The test verifies that the solution satisfies the perturbed equation and the boundary condition.
    '''
    from pySDC.helpers.spectral_helper import SpectralHelper
    import numpy as np
    import scipy.sparse as sp

    helper = SpectralHelper()
    helper.add_component('u')
    helper.add_axis(base='cheby', N=N)
    helper.setup_fft()

    helper.add_BC('u', 'u', 0, bc, bc_val)
    helper.setup_BCs()

    C = helper.get_basis_change_matrix()
    D = helper.get_differentiation_matrix(axes=(-1,))
    Id = helper.get_Id()

    _A = helper.get_empty_operator_matrix()
    helper.add_equation_lhs(_A, 'u', {'u': D - Id})
    A = helper.convert_operator_matrix_to_operator(_A)
    A = helper.put_BCs_in_matrix(A)

    rhs = helper.put_BCs_in_rhs(np.zeros((1, N)))
    rhs_hat = helper.transform(rhs, axes=(-1,))

    sol_hat = sp.linalg.spsolve(A, rhs_hat.flatten())

    x = helper.get_grid()
    coef = np.append(np.zeros(N - 1), [1])
    P = np.polynomial.Chebyshev(C @ coef)

    sol_poly = np.polynomial.Chebyshev(sol_hat)
    d_sol_poly = sol_poly.deriv(1)
    x = np.linspace(-1, 1, 100)

    assert np.isclose(sol_poly(bc), bc_val), 'Solution does not satisfy boundary condition'

    tau = (d_sol_poly(x) - sol_poly(x)) / P(x)
    assert np.allclose(tau, tau[0]), 'Solution does not satisfy perturbed equation'


@pytest.mark.base
@pytest.mark.parametrize('mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
@pytest.mark.parametrize('bc_val', [-2, 1.0])
def test_tau_method2D(mode, nz, nx, bc_val, bc=-1, plotting=False):
    '''
    solve u_z - 0.1u_xx -u_x + tau P = 0, u(bc) = sin(bc_val*x) -> space-time discretization of advection-diffusion
    problem. We do FFT in x-direction and Chebychov in z-direction.
    '''
    from pySDC.helpers.spectral_helper import SpectralHelper
    import numpy as np

    helper = SpectralHelper()
    helper.add_axis('fft', N=nx)
    helper.add_axis('cheby', N=nz, mode=mode)
    helper.add_component(['u'])
    helper.setup_fft()

    x = helper.axes[0].get_1dgrid()
    z = helper.axes[1].get_1dgrid()
    Z, X = helper.get_grid()

    bcs = np.sin(bc_val * x)
    helper.add_BC('u', 'u', 1, bc, bcs)
    helper.setup_BCs()

    # generate matrices
    Dz = helper.get_differentiation_matrix(axes=(1,))
    Dx = helper.get_differentiation_matrix(axes=(0,))
    Dxx = helper.get_differentiation_matrix(axes=(0,), p=2)

    # generate operator
    _A = helper.get_empty_operator_matrix()
    helper.add_equation_lhs(_A, 'u', {'u': Dz - Dxx * 1e-1 - Dx})
    A = helper.convert_operator_matrix_to_operator(_A)

    # prepare system to solve
    A = helper.put_BCs_in_matrix(A)
    rhs = helper.put_BCs_in_rhs(np.zeros(((1,) + X.shape)))
    rhs_hat = helper.transform(rhs, axes=(-1, -2))

    # solve the system
    sol_hat = helper.u_init
    sol_hat[0] = (helper.sparse_lib.linalg.spsolve(A, rhs_hat.flatten())).reshape(X.shape)
    sol = helper.itransform(sol_hat, axes=(-2, -1)).real

    # construct polynomials for testing
    sol_cheby = helper.itransform(sol_hat, axes=(-2,))
    polys = [np.polynomial.Chebyshev(sol_cheby[0, i, :]) for i in range(nx)]

    Pz = np.polynomial.Chebyshev(np.append(np.zeros(nz - 1), [1]))
    tau_term, _ = np.meshgrid(Pz(z), np.ones(nx))
    error = ((A @ sol_hat.flatten()).reshape(X.shape) / tau_term).real

    if plotting:
        import matplotlib.pyplot as plt

        im = plt.pcolormesh(X, Z, sol[0])
        # im = plt.pcolormesh(X, Z, error)
        plt.colorbar(im)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()

    for i in range(nx):

        assert np.isclose(polys[i](bc), bcs[i]), f'Solution does not satisfy boundary condition x={x[i]}'

        assert np.allclose(
            polys[i](z), sol[0, i, :]
        ), f'Solution is incorrectly transformed back to real space at x={x[i]}'

    assert np.allclose(error, error[0, 0]), 'Solution does not satisfy perturbed equation'


if __name__ == '__main__':
    str_to_bool = lambda me: False if me == 'False' else True
    str_to_tuple = lambda arg: tuple(int(me) for me in arg.split(','))

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, help='Dof in x direction')
    parser.add_argument('--nz', type=int, help='Dof in z direction')
    parser.add_argument('--axes', type=str_to_tuple, help='Axes over which to transform')
    parser.add_argument('--bz', type=str, help='Base in z direction')
    parser.add_argument('--bx', type=str, help='Base in x direction')
    parser.add_argument('--test', type=str, help='type of test', choices=['transform'])
    parser.add_argument('--useMPI', type=str_to_bool, help='use MPI or not', choices=[True, False], default=True)
    args = parser.parse_args()

    if args.test == 'transform':
        test_transform(**vars(args))
    elif args.test is None:
        test_transform(4, 3, 'cheby', False)
        # test_transform_MPI(4, 3, 'cheby')
        # test_differentiation_matrix2D(2, 2, 'T2U', (-1,))
        # test_matrix1D(4, 'cheby', 'int')
        # test_tau_method(-1, 8, -1)
        # test_tau_method2D('T2U', 2**7, 2**5, -2, plotting=True)
    else:
        raise NotImplementedError
    print('done')
