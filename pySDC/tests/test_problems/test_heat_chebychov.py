import pytest


@pytest.mark.base
@pytest.mark.parametrize('a', [-2, 19])
@pytest.mark.parametrize('b', [-7.3, 66])
@pytest.mark.parametrize('f', [0, 1])
def test_heat1d_chebychov(a, b, f, nvars=2**4):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat1DChebychov

    P = Heat1DChebychov(nvars=nvars, a=a, b=b, f=f, nu=1e-3)

    u0 = P.u_exact(0)
    dt = 1e-1
    u = P.solve_system(u0, dt)
    u02 = u - dt * P.eval_f(u)

    assert np.allclose(u, P.u_exact(dt), atol=1e-8), 'Error in solver'
    assert np.allclose(u0[0], u02[0], atol=1e-8), 'Error in eval_f'


@pytest.mark.base
@pytest.mark.parametrize('a', [0, 7])
@pytest.mark.parametrize('b', [0])
@pytest.mark.parametrize('c', [0, 3.1415])
@pytest.mark.parametrize('fx', [2, 1])
@pytest.mark.parametrize('fy', [2, 1])
@pytest.mark.parametrize('base_x', ['fft', 'chebychov'])
@pytest.mark.parametrize('base_y', ['fft', 'chebychov'])
def test_heat2d_chebychov(a, b, c, fx, fy, base_x, base_y, nx=2**5, ny=2**5):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat2DChebychov

    if base_y == 'fft' and (a != 0 or c != 0):
        return None
    if base_x == 'fft' and (b != 0):
        return None

    P = Heat2DChebychov(nx=nx, ny=ny, a=a, b=b, c=c, fx=fx, fy=fy, base_x=base_x, base_y=base_y, nu=1e-3)

    u0 = P.u_exact(0)
    dt = 1e-1
    u = P.solve_system(u0, dt)
    u02 = u - dt * P.eval_f(u)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    idx = 0
    im = axs[0].pcolormesh(P.X, P.Y, (u[idx] - P.u_exact(dt)[idx]).real)
    im2 = axs[1].pcolormesh(P.X, P.Y, (u0 - u02)[idx].real)
    fig.colorbar(im)
    fig.colorbar(im2)
    # plt.show()

    print(abs((u - P.u_exact(dt))))
    print(abs(u0[0] - u02[0]))

    assert np.allclose(u, P.u_exact(dt), atol=1e-4), 'Error in solver'
    assert np.allclose(u0[0], u02[0], atol=1e-3), 'Error in eval_f'


@pytest.mark.base
@pytest.mark.parametrize('mode', ['T2U', 'D2U'])
@pytest.mark.parametrize('preconditioning', [True, False])
def test_heat1d_chebychov_preconditioning(mode, preconditioning, plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat1DChebychovPreconditioning
    import scipy
    from pySDC.helpers.problem_helper import ChebychovHelper

    N = 2**2
    P = Heat1DChebychovPreconditioning(
        nvars=N,
        a=-2,
        b=3,
        poly_coeffs=[-1, 1],
        mode=mode,
        nu=1e-1,
        preconditioning=preconditioning,
    )
    cheby = ChebychovHelper(N)

    u0 = P.u_exact()
    u_hat = P.cheby.transform(u0)

    dt = 1e-1
    sol = P.solve_system(rhs=u0, factor=dt, u0=u0)
    sol_hat = P.cheby.transform(sol)

    # for computing forward Euler, we need to reevaluate the spatial derivative
    backward = sol - dt * P.eval_f(sol)
    backward[P.idu] = P._compute_derivative(backward[P.iu])

    deriv = P._compute_derivative(u0[P.idu])

    if plot:
        import matplotlib.pyplot as plt

        for i in [P.iu]:
            plt.plot(P.x, u0[i], label=f'u0[{i}]')
            plt.plot(P.x, sol[i], ls='--', label=f'BE[{i}]')
            plt.plot(P.x, backward[i], ls='-.', label=f'BFE[{i}]')
            # plt.plot(P.x, deriv)
        plt.legend(frameon=False)
        plt.show()

    # test that the solution satisfies the algebraic constraints
    D_u = P._compute_derivative(sol[P.iu])
    assert np.allclose(D_u, sol[P.idu]), 'The solution of backward Euler does not satisfy the algebraic constraints'

    # # test that f evaluation is second derivative of the solution
    # f = P.eval_f(u0)
    # D2 = cheby.get_T2T_differentiation_matrix(2)
    # D1 = cheby.get_T2T_differentiation_matrix(1)
    # D2u = scipy.fft.itransform(D2 @ u_hat[P.iu] / P.norm)
    # D1u = scipy.fft.itransform(D1 @ u_hat[P.iu] / P.norm)
    # assert np.allclose(D2u, f[P.iu]), 'The time derivative is not the second space derivative.'
    # assert np.allclose(
    #     D1u, u0[P.idu]
    # ), 'The initial conditions don\'t have the first space derivative where it needs to be.'

    assert np.allclose(
        u0[P.iu], P.solve_system(u0, 1e-9, u0)[P.iu], atol=1e-7
    ), 'We did not get back the initial conditions when solving with \"zero\" step size.'
    assert np.allclose(u0[P.iu], backward[P.iu], atol=1e-7), abs(u0 - backward)


def test_SDC(plotting=False):
    import numpy as np
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat1DChebychovPreconditioning
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    dt = 1
    Tend = 10

    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1e-9

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'

    problem_params = {'a': -9, 'b': 3, 'poly_coeffs': (0, 0, 0, -1, 1), 'nvars': 2**5, 'mode': 'D2U'}

    step_params = {}
    step_params['maxiter'] = 1

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = []

    description = {}
    description['problem_class'] = Heat1DChebychovPreconditioning
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)

    a, b = problem_params['a'], problem_params['b']
    expect = (b - a) / 2 * P.x + (b + a) / 2
    if plotting:
        import matplotlib.pyplot as plt

        x = P.x
        i = P.iu
        plt.plot(x, uinit[i])
        plt.plot(x, uend[i])
        plt.plot(P.x, expect)
        plt.show()
    assert np.allclose(uend[0], expect)


@pytest.mark.base
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
@pytest.mark.parametrize('mode', ['T2T', 'T2U'])
def test_heat2d(mode, nx, nz, plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat2d

    P = Heat2d(nx=nx, nz=nz, nu=1e-2, a=-1, b=2, bc_type='dirichlet', mode='T2U')

    u0 = P.u_exact()

    u0_hat = P.transform(u0)
    assert np.allclose(u0, P.itransform(u0_hat))

    dt = 3.0e-1
    un = P.solve_system(u0, dt)

    # un[1] = P._compute_derivative(un[0])
    # u02 = un - dt * P.eval_f(un)
    # err = u0 - u02
    # print(np.max(abs(err)))

    # solve something akin to a Poisson problem
    dtP = 1e9
    uP = P.solve_system(u0 * 0, dtP)
    uP_expect = (P.b - P.a) / 2 * P.Z + (P.b + P.a) / 2.0

    if plot:

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        args = {'vmin': P.a * 1.2, 'vmax': P.b * 1.2}
        axs[0].pcolormesh(P.X, P.Z, u0[0], **args)
        axs[1].pcolormesh(P.X, P.Z, un[0], **args)
        # axs[0].pcolormesh(P.X, P.Z, u02[0], **args)

        idx = nx // 3
        axs[2].plot(P.Z[idx], u0[0][idx])
        axs[2].plot(P.Z[idx], un[0][idx])
        axs[2].set_title('z-direction')

        idx = nz // 2
        axs[3].plot(P.X[:, idx], u0[0][:, idx])
        axs[3].plot(P.X[:, idx], un[0][:, idx])
        axs[3].set_title('x-direction')
        plt.show()

    assert np.allclose(uP[0], uP_expect, atol=1e3 / dtP), 'Got unexpected solution of Poisson problem!'


if __name__ == '__main__':
    # test_SDC(True)
    # test_heat1d_chebychov('T2U', False, plot=True)
    # test_heat2d('T2T', 2**4, 2**5, True)
    # test_AdvectionDiffusion(plot=True)
    # test_heat1d_chebychov_preconditioning('D2U', True)
    test_heat2d_chebychov(0, 0, 0, 2, 2, 'chebychov', 'fft', 2**6, 2**6)
