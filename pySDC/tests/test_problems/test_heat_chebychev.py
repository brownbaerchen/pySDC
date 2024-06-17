import pytest


@pytest.mark.base
@pytest.mark.parametrize('mode', ['T2U', 'D2U'])
@pytest.mark.parametrize('preconditioning', [True, False])
def test_heat1d_chebychev(mode, preconditioning, plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import Heat1DChebychev
    import scipy
    from pySDC.helpers.problem_helper import ChebychovHelper

    N = 2**6
    P = Heat1DChebychev(
        nvars=N,
        a=-2,
        b=3,
        poly_coeffs=[0, 0, 0, -1, 1],
        solver_type='direct',
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
    from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import Heat1DChebychev
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
    description['problem_class'] = Heat1DChebychev
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
def test_heat2d(plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import Heat2d

    nx = 2**5
    nz = 2**6
    P = Heat2d(nx=nx, nz=nz, a=-1, b=2)

    u0 = P.u_exact()

    u0_hat = P.transform(u0)
    assert np.allclose(u0, P.itransform(u0_hat))

    dt = 3.0e-2
    un = P.solve_system(u0, dt)
    un = u0 + dt * P.eval_f(u0)

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].pcolormesh(P.X, P.Z, u0[0])
        axs[1].pcolormesh(P.X, P.Z, un[0])
        idx = nx // 2
        print(u0.shape)
        axs[2].plot(P.Z[idx], u0[0][idx])
        axs[2].plot(P.Z[idx], un[0][idx])

        idx = nz // 2
        axs[3].plot(P.X[:, idx], u0[0][:, idx])
        axs[3].plot(P.X[:, idx], un[0][:, idx])
        plt.show()


@pytest.mark.base
def test_AdvectionDiffusion(plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import AdvectionDiffusion

    nx = 2**0
    nz = 2**1
    P = AdvectionDiffusion(nx=nx, nz=nz, a=1.0, b=1.0, nu=1.0e-1, c=-0.0e0)

    u0 = P.u_exact()

    u0_hat = P.transform(u0)
    assert np.allclose(u0, P.itransform(u0_hat))

    dt = 1.0e-1
    un = P.solve_system(u0, dt)
    f = P.eval_f(u0)
    # un = u0 + dt * P.eval_f(u0)

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].pcolormesh(P.X, P.Z, u0[0])
        # axs[1].pcolormesh(P.X, P.Z, f[0])
        # axs[1].pcolormesh(P.X, P.Z, u0[1])
        axs[1].pcolormesh(P.X, P.Z, un[0])
        idx = nx // 4
        axs[2].plot(P.Z[idx], u0[0][idx], marker='.')
        axs[2].plot(P.Z[idx], un[0][idx], '--', marker='.')

        idx = nz // 4
        axs[3].plot(P.X[:, idx], u0[0][:, idx])
        axs[3].plot(P.X[:, idx], un[0][:, idx], '--')
        plt.show()


if __name__ == '__main__':
    # test_heat1d_chebychev('T2U', False, plot=True)
    test_AdvectionDiffusion(plot=True)
