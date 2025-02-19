import pytest


@pytest.mark.base
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('ny', [16])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
@pytest.mark.parametrize('axes', [(-2,), (-1,), (-2, -1)])
def test_integration_matrix2D(nx, ny, variant, axes, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base='fft', N=nx)
    helper.add_axis(base='cheby', N=ny)
    helper.setup_fft()

    X, Z = helper.get_grid()

    conv = helper.get_basis_change_matrix()
    S = helper.get_integration_matrix(axes=axes)

    u = helper.u_init
    u[0, ...] = np.sin(X) * Z**2 + np.cos(X) * Z**3
    if axes == (-2,):
        expect = -np.cos(X) * Z**2 + np.sin(X) * Z**3
    elif axes == (-1,):
        expect = np.sin(X) * 1 / 3 * Z**3 + np.cos(X) * 1 / 4 * Z**4
    elif axes in [(-2, -1), (-1, -2)]:
        expect = -np.cos(X) * 1 / 3 * Z**3 + np.sin(X) * 1 / 4 * Z**4
    else:
        raise NotImplementedError

    u_hat = helper.transform(u, axes=(-2, -1))
    S_u_hat = (conv @ S @ u_hat.flatten()).reshape(u_hat.shape)
    S_u = helper.itransform(S_u_hat, axes=(-1, -2))

    assert np.allclose(S_u, expect, atol=1e-12)


@pytest.mark.base
@pytest.mark.parametrize('nx', [32])
@pytest.mark.parametrize('ny', [16])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
@pytest.mark.parametrize('axes', [(-2,), (-1,), (-2, -1)])
@pytest.mark.parametrize('bx', ['cheby', 'fft'])
@pytest.mark.parametrize('by', ['cheby', 'fft'])
def test_differentiation_matrix2D(nx, ny, variant, axes, bx, by, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=by, N=ny)
    helper.setup_fft()

    X, Z = helper.get_grid()
    conv = helper.get_basis_change_matrix()
    D = helper.get_differentiation_matrix(axes)

    u = helper.u_init

    if by == 'cheby':
        u[0, ...] = np.sin(X) * Z**2 + Z**3 + np.cos(2 * X)
        if axes == (-2,):
            expect = np.cos(X) * Z**2 - 2 * np.sin(2 * X)
        elif axes == (-1,):
            expect = np.sin(X) * Z * 2 + Z**2 * 3
        elif axes in [(-2, -1), (-1, -2)]:
            expect = np.cos(X) * 2 * Z
        else:
            raise NotImplementedError
    else:
        u[0, ...] = np.sin(X) * np.cos(2 * Z) + np.cos(2 * X) + np.sin(Z)
        if axes == (-2,):
            expect = np.cos(X) * np.cos(2 * Z) - 2 * np.sin(2 * X)
        elif axes == (-1,):
            expect = np.sin(X) * (-2) * np.sin(2 * Z) + np.cos(Z)
        elif axes in [(-2, -1), (-1, -2)]:
            expect = -2 * np.cos(X) * np.sin(2 * Z)
        else:
            raise NotImplementedError

    u_hat = helper.transform(u, axes=(-2, -1))
    D_u_hat = (conv @ D @ u_hat.flatten()).reshape(u_hat.shape)
    D_u = helper.itransform(D_u_hat, axes=(-1, -2)).real

    assert np.allclose(D_u, expect, atol=1e-12)


@pytest.mark.base
@pytest.mark.parametrize('nx', [32])
@pytest.mark.parametrize('ny', [0, 16])
@pytest.mark.parametrize('nz', [8])
@pytest.mark.parametrize('variant', ['T2U', 'T2T'])
@pytest.mark.parametrize('bx', ['cheby', 'fft'])
def test_identity_matrix_ND(nx, ny, nz, variant, bx, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base=bx, N=nx)
    if ny > 0:
        helper.add_axis(base=bx, N=ny)
    helper.add_axis(base='cheby', N=nz)
    helper.setup_fft()

    grid = helper.get_grid()
    conv = helper.get_basis_change_matrix()
    I = helper.get_Id()

    u = helper.u_init
    u[0, ...] = np.sin(grid[-2]) * grid[-1] ** 2 + grid[-1] ** 3 + np.cos(2 * grid[-2])

    if ny > 0:
        u[0, ...] += np.cos(grid[-3])

    u_hat = helper.transform(u, axes=(-2, -1))
    I_u_hat = (conv @ I @ u_hat.flatten()).reshape(u_hat.shape)
    I_u = helper.itransform(I_u_hat, axes=(-1, -2))

    assert np.allclose(I_u, u, atol=1e-12)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('base', ['cheby'])
@pytest.mark.parametrize('type', ['diff', 'int'])
def test_matrix1D(N, base, type):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    coeffs = np.random.random(N)

    helper = SpectralHelper(debug=True)
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


def _test_transform_dealias(
    bx,
    by,
    axis,
    nx=2**4 + 1,
    ny=2**2 + 1,
    padding=3 / 2,
    axes=(
        -2,
        -1,
    ),
    useMPI=True,
    **kwargs,
):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=by, N=ny)
    helper.setup_fft()
    xp = helper.xp

    _padding = [
        padding,
    ] * helper.ndim

    helper_pad = SpectralHelper(comm=comm, debug=True)
    helper_pad.add_axis(base=bx, N=int(_padding[0] * nx))
    helper_pad.add_axis(base=by, N=int(_padding[1] * ny))
    helper_pad.setup_fft()

    u_hat = helper.u_init_forward
    u2_hat_expect = helper.u_init_forward
    u_expect = helper.u_init
    u_expect_pad = helper_pad.u_init
    Kx, Kz = helper.get_wavenumbers()
    X, Z = helper.get_grid()
    X_pad, Z_pad = helper_pad.get_grid()

    if axis == -2:
        f = nx // 3
        u_hat[0][xp.logical_and(xp.abs(Kx) == f, Kz == 0)] += 1
        u2_hat_expect[0][xp.logical_and(xp.abs(Kx) == 2 * f, Kz == 0)] += 1 / nx
        u2_hat_expect[0][xp.logical_and(xp.abs(Kx) == 0, Kz == 0)] += 2 / nx
        u_expect[0] += np.cos(f * X) * 2 / nx
        u_expect_pad[0] += np.cos(f * X_pad) * 2 / nx
    elif axis == -1:
        f = ny // 2 + 1
        u_hat[0][xp.logical_and(xp.abs(Kz) == f, Kx == 0)] += 1
        u2_hat_expect[0][xp.logical_and(Kz == 2 * f, Kx == 0)] += 1 / (2 * nx)
        u2_hat_expect[0][xp.logical_and(Kz == 0, Kx == 0)] += 1 / (2 * nx)

        coef = np.zeros(ny)
        coef[f] = 1 / nx
        u_expect[0] = np.polynomial.Chebyshev(coef)(Z)
        u_expect_pad[0] = np.polynomial.Chebyshev(coef)(Z_pad)
    elif axis in [(-1, -2), (-2, -1)]:
        fx = nx // 3
        fz = ny // 2 + 1

        u_hat[0][xp.logical_and(xp.abs(Kx) == fx, Kz == 0)] += 1
        u_hat[0][xp.logical_and(xp.abs(Kz) == fz, Kx == 0)] += 1

        u2_hat_expect[0][xp.logical_and(xp.abs(Kx) == 2 * fx, Kz == 0)] += 1 / nx
        u2_hat_expect[0][xp.logical_and(xp.abs(Kx) == 0, Kz == 0)] += 2 / nx
        u2_hat_expect[0][xp.logical_and(Kz == 2 * fz, Kx == 0)] += 1 / (2 * nx)
        u2_hat_expect[0][xp.logical_and(Kz == 0, Kx == 0)] += 1 / (2 * nx)
        u2_hat_expect[0][xp.logical_and(Kz == fz, xp.abs(Kx) == fx)] += 2 / nx

        coef = np.zeros(ny)
        coef[fz] = 1 / nx

        u_expect[0] = np.cos(fx * X) * 2 / nx + np.polynomial.Chebyshev(coef)(Z)
        u_expect_pad[0] = np.cos(fx * X_pad) * 2 / nx + np.polynomial.Chebyshev(coef)(Z_pad)
    else:
        raise NotImplementedError

    assert bx == 'fft' and by == 'cheby', 'This test is not implemented for the bases you are looking for'

    u_pad = helper.itransform(u_hat, padding=_padding, axes=axes)
    u = helper.itransform(u_hat, axes=axes).real

    assert not np.allclose(u_pad.shape, u.shape)

    u2 = u**2
    u2_pad = u_pad**2

    assert np.allclose(u, u_expect)
    assert np.allclose(u_pad, u_expect_pad)

    assert np.allclose(u2_hat_expect, helper.transform(u2_pad, padding=_padding))
    assert not np.allclose(u2_hat_expect, helper.transform(u2)), 'Test is too boring, no dealiasing needed'


@pytest.mark.base
@pytest.mark.parametrize('nx', [3, 8])
@pytest.mark.parametrize('ny', [3, 8])
@pytest.mark.parametrize('nz', [0, 3, 8])
@pytest.mark.parametrize('by', ['fft', 'cheby'])
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
@pytest.mark.parametrize('bx', ['fft', 'cheby'])
@pytest.mark.parametrize('axes', [(-1,), (-2,), (-3,), (-1, -2), (-2, -1), (-1, -2, -3)])
def test_transform(nx, ny, nz, bx, by, bz, axes, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper
    from pytest_mpi import parallel_assert

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=by, N=ny)
    if nz > 0:
        helper.add_axis(base=bz, N=nz)
    elif -3 in axes:
        return None
    helper.setup_fft()

    u = helper.u_init
    u[...] = np.random.random(u.shape)

    if nz > 0:
        u_all = np.empty(shape=(1, nx, ny, nz), dtype=u.dtype)
    else:
        u_all = np.empty(shape=(1, nx, ny), dtype=u.dtype)

    if useMPI:
        u_all[...] = (np.concatenate(comm.allgather(u[0]), axis=0)).reshape(u_all.shape)
        if comm.size == 1:
            assert np.allclose(u_all, u)
    else:
        u_all[...] = u

    axes_ordered = []
    for ax in axes:
        if 'FFT' in type(helper.axes[ax]).__name__:
            axes_ordered = axes_ordered + [ax]
        else:
            axes_ordered = [ax] + axes_ordered

    expect_trf = u_all.copy()
    for i in axes_ordered:
        base = helper.axes[i]
        expect_trf = base.transform(expect_trf, axis=i)

    trf = helper.transform(u, axes=axes)
    itrf = helper.itransform(trf, axes=axes)

    expect_local = expect_trf[
        (
            ...,
            *helper.local_slice,
        )
    ]

    parallel_assert(lambda: np.allclose(expect_local, trf), msg='Forward transform is unexpected')
    parallel_assert(lambda: np.allclose(itrf, u), msg='Backward transform is unexpected')


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
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize('nx', [3, 8])
@pytest.mark.parametrize('ny', [3, 8])
@pytest.mark.parametrize('nz', [0, 8])
@pytest.mark.parametrize(
    'by',
    [
        'fft',
    ],
)
@pytest.mark.parametrize(
    'bx',
    [
        'fft',
    ],
)
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
@pytest.mark.parametrize('axes', [(-1,), (-1, -2), (-2, -1, -3)])
def test_transform_MPI(nx, ny, nz, bx, by, bz, axes, **kwargs):
    test_transform(nx=nx, ny=ny, nz=nz, bx=bx, by=by, bz=bz, axes=axes, useMPI=True, **kwargs)


# @pytest.mark.parametrize('num_procs', [2, 1])
# @pytest.mark.parametrize('axes', ["-1", "-2", "-1,-2"])
# def test_transform_MPI(nx, ny, bx, by, num_procs, axes):
#     run_MPI_test(num_procs=num_procs, test='transform', nx=nx, ny=ny, bx=bx, by=by, axes=axes)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [8])
@pytest.mark.parametrize('ny', [16])
@pytest.mark.parametrize('bx', ['fft'])
@pytest.mark.parametrize('num_procs', [2, 1])
@pytest.mark.parametrize('axes', ["-1", "-1,-2"])
def test_differentiation_MPI(nx, ny, bx, num_procs, axes):
    run_MPI_test(num_procs=num_procs, test='diff', nx=nx, ny=ny, bx=bx, by='cheby', axes=axes)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [8])
@pytest.mark.parametrize('ny', [16])
@pytest.mark.parametrize('num_procs', [2, 1])
@pytest.mark.parametrize('axes', ["-1", "-1,-2"])
def test_integration_MPI(nx, ny, num_procs, axes):
    run_MPI_test(num_procs=num_procs, test='int', nx=nx, ny=ny, axes=axes)


@pytest.mark.base
@pytest.mark.parametrize('N', [8, 32])
@pytest.mark.parametrize('bc_val', [0, 3.1415])
def test_tau_method_integral(N, bc_val):
    test_tau_method(bc=0, N=N, bc_val=bc_val, kind='integral')


@pytest.mark.base
@pytest.mark.parametrize('bc', [-1, 0, 1])
@pytest.mark.parametrize('N', [8, 32])
@pytest.mark.parametrize('bc_val', [-99, 3.1415])
def test_tau_method(bc, N, bc_val, kind='Dirichlet'):
    '''
    solve u_x - u + tau P = 0, u(bc) = bc_val

    We choose P = T_N or U_N. We replace the last row in the matrix with the boundary condition to get a
    unique solution for the given resolution.

    The test verifies that the solution satisfies the perturbed equation and the boundary condition.
    '''
    from pySDC.helpers.spectral_helper import SpectralHelper
    import numpy as np
    import scipy.sparse as sp

    helper = SpectralHelper(debug=True)
    helper.add_component('u')
    helper.add_axis(base='cheby', N=N)
    helper.setup_fft()

    if kind == 'integral':
        helper.add_BC('u', 'u', 0, v=bc_val, kind='integral')
    else:
        helper.add_BC('u', 'u', 0, x=bc, v=bc_val, kind='Dirichlet')
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

    sol_poly = np.polynomial.Chebyshev(sol_hat)
    d_sol_poly = sol_poly.deriv(1)
    x = np.linspace(-1, 1, 100)

    if kind == 'integral':
        integral = sol_poly.integ(1, lbnd=-1)
        assert np.isclose(integral(1), bc_val), 'Solution does not satisfy boundary condition'
    else:
        assert np.isclose(sol_poly(bc), bc_val), 'Solution does not satisfy boundary condition'

    coef = np.append(np.zeros(N - 1), [1])
    P = np.polynomial.Chebyshev(C @ coef)
    tau = (d_sol_poly(x) - sol_poly(x)) / P(x)
    assert np.allclose(tau, tau[0]), 'Solution does not satisfy perturbed equation'


@pytest.mark.base
@pytest.mark.parametrize('variant', ['T2T', 'T2U'])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('ny', [4, 8])
@pytest.mark.parametrize('bc_val', [-2, 1.0])
def test_tau_method2D(variant, ny, nx, bc_val, bc=-1, useMPI=False, plotting=False, **kwargs):
    '''
    solve u_z - 0.1u_xx -u_x + tau P = 0, u(bc) = sin(bc_val*x) -> space-time discretization of advection-diffusion
    problem. We do FFT in x-direction and Chebychov in z-direction.
    '''
    from pySDC.helpers.spectral_helper import SpectralHelper
    import numpy as np

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis('fft', N=nx)
    helper.add_axis('cheby', N=ny)
    helper.add_component(['u'])
    helper.setup_fft()

    X, Z = helper.get_grid()
    x = X[:, 0]
    z = Z[0, :]
    shape = helper.init[0][1:]

    bcs = np.sin(bc_val * x)
    helper.add_BC('u', 'u', 1, kind='dirichlet', x=bc, v=bcs)
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
    rhs_hat = helper.put_BCs_in_rhs_hat(helper.u_init_forward)

    # solve the system
    sol_hat = helper.u_init_forward
    sol_hat[0] = (helper.sparse_lib.linalg.spsolve(A, rhs_hat.flatten())).reshape(X.shape)
    sol = helper.itransform(sol_hat, axes=(-2, -1)).real

    # construct polynomials for testing
    sol_cheby = helper.transform(sol, axes=(-1,))
    polys = [np.polynomial.Chebyshev(sol_cheby[0, i, :]) for i in range(shape[0])]

    Pz = np.polynomial.Chebyshev(np.append(np.zeros(ny - 1), [1]))
    tau_term, _ = np.meshgrid(Pz(z), np.ones(shape[0]))
    error = ((A @ sol_hat.flatten()).reshape(X.shape) / tau_term).real

    if plotting:
        import matplotlib.pyplot as plt

        im = plt.pcolormesh(X, Z, sol[0])
        plt.colorbar(im)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()

    for i in range(shape[0]):

        assert np.isclose(polys[i](bc), bcs[i]), f'Solution does not satisfy boundary condition x={x[i]}'

        assert np.allclose(
            polys[i](z), sol[0, i, :]
        ), f'Solution is incorrectly transformed back to real space at x={x[i]}'

    assert np.allclose(error, error[0, 0]), 'Solution does not satisfy perturbed equation'


@pytest.mark.mpi4py
@pytest.mark.parametrize('variant', ['T2T', 'T2U'])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('ny', [4, 8])
@pytest.mark.parametrize('bc_val', [-2])
@pytest.mark.parametrize('num_procs', [2, 1])
def test_tau_method2D_MPI(variant, ny, nx, bc_val, num_procs, **kwargs):
    run_MPI_test(variant=variant, ny=ny, nx=nx, bc_val=bc_val, num_procs=num_procs, test='tau')


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_procs', [1])
@pytest.mark.parametrize('axis', [-1, -2])
@pytest.mark.parametrize('bx', ['fft'])
@pytest.mark.parametrize('by', ['cheby'])
def test_dealias_MPI(num_procs, axis, bx, by, nx=32, ny=64, **kwargs):
    run_MPI_test(num_procs=num_procs, axis=axis, nx=nx, ny=ny, bx=bx, by=by, test='dealias')


if __name__ == '__main__':
    str_to_bool = lambda me: False if me == 'False' else True
    str_to_tuple = lambda arg: tuple(int(me) for me in arg.split(','))

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, help='Dof in x direction')
    parser.add_argument('--ny', type=int, help='Dof in y direction')
    parser.add_argument('--axes', type=str_to_tuple, help='Axes over which to transform')
    parser.add_argument('--axis', type=int, help='Direction of the action')
    parser.add_argument('--by', type=str, help='Base in y direction')
    parser.add_argument('--bx', type=str, help='Base in x direction')
    parser.add_argument('--bc_val', type=int, help='Value of boundary condition')
    parser.add_argument('--test', type=str, help='type of test', choices=['transform', 'diff', 'int', 'tau', 'dealias'])
    parser.add_argument('--variant', type=str, help='Chebychov mode', choices=['T2T', 'T2U'], default='T2U')
    parser.add_argument('--useMPI', type=str_to_bool, help='use MPI or not', choices=[True, False], default=True)
    args = parser.parse_args()

    if args.test == 'transform':
        test_transform(**vars(args))
    elif args.test == 'diff':
        test_differentiation_matrix2D(**vars(args))
    elif args.test == 'int':
        test_integration_matrix2D(**vars(args))
    elif args.test == 'tau':
        test_tau_method2D(**vars(args))
    elif args.test == 'dealias':
        _test_transform_dealias(**vars(args))
    elif args.test is None:
        test_transform_MPI(3, 4, 3, 'fft', 'fft', 'fft', (-1,))
        # test_identity_matrix_ND(2, 1, 4, 'T2U', 'fft')
        # test_differentiation_matrix2D(2**5, 2**5, 'T2U', bx='fft', by='fft', axes=(-2, -1))
        # test_matrix1D(4, 'cheby', 'int')
        # test_tau_method(-1, 8, 99, kind='Dirichlet')
        # test_tau_method2D('T2U', 2**8, 2**8, -2, plotting=True)
        # test_filter(6, 6, (0,))
        # _test_transform_dealias('fft', 'cheby', (-1, -2))
    else:
        raise NotImplementedError
    print('done')
