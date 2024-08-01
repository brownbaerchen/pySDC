import pytest


@pytest.mark.base
@pytest.mark.parametrize('a', [-2, 19])
@pytest.mark.parametrize('b', [-7.3, 66])
@pytest.mark.parametrize('f', [0, 1])
def test_heat1d_chebychov(a, b, f, nvars=2**4):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat1DChebychov

    P = Heat1DChebychov(nvars=nvars, a=a, b=b, f=f, nu=1e-3, left_preconditioner=False, right_preconditioning='T2T')

    u0 = P.u_exact(0)
    dt = 1e-1
    u = P.solve_system(u0, dt)
    u02 = u - dt * P.eval_f(u)

    assert np.allclose(u, P.u_exact(dt), atol=1e-8), 'Error in solver'
    assert np.allclose(u0[0], u02[0], atol=1e-8), 'Error in eval_f'


@pytest.mark.base
@pytest.mark.parametrize('a', [0, 7])
@pytest.mark.parametrize('b', [0, -2.77])
@pytest.mark.parametrize('c', [0, 3.1415])
@pytest.mark.parametrize('fx', [2, 1])
@pytest.mark.parametrize('fy', [2, 1])
@pytest.mark.parametrize('base_x', ['fft', 'chebychov'])
@pytest.mark.parametrize('base_y', ['fft', 'chebychov'])
def test_heat2d_chebychov(a, b, c, fx, fy, base_x, base_y, nx=2**5, ny=2**5):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat2DChebychov

    if base_y == 'fft' and (b != c):
        return None
    if base_x == 'fft' and (b != a):
        return None

    P = Heat2DChebychov(
        nx=nx,
        ny=ny,
        a=a,
        b=b,
        c=c,
        fx=fx,
        fy=fy,
        base_x=base_x,
        base_y=base_y,
        nu=1e-3,
        left_preconditioner=False,
        right_preconditioning='T2T',
    )

    u0 = P.u_exact(0)
    dt = 1e-1
    u = P.solve_system(u0, dt)
    u02 = u - dt * P.eval_f(u)

    tol = 1e-6 if (base_x == 'fft' or base_y == 'fft') else 1e-3

    assert np.allclose(
        u, P.u_exact(dt), atol=tol
    ), f'Error in solver larger than expected, got {abs((u - P.u_exact(dt))):.2e}'
    assert np.allclose(u0[0], u02[0], atol=1e-10), 'Error in eval_f'


def test_SDC():
    import numpy as np
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat1DChebychov
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.hooks.log_work import LogSDCIterations

    generic_implicit.compute_residual = compute_residual_DAE

    dt = 1e-1
    Tend = 2 * dt

    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1e-10

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 4
    sweeper_params['QI'] = 'LU'

    problem_params = {}

    step_params = {}
    step_params['maxiter'] = 99

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogSDCIterations]

    description = {}
    description['problem_class'] = Heat1DChebychov
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t=0)

    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)
    u_exact = P.u_exact(t=Tend)
    assert np.allclose(uend, u_exact, atol=1e-10)

    k = get_sorted(stats, type='k')
    assert all(me[1] < step_params['maxiter'] for me in k)


if __name__ == '__main__':
    test_SDC()
    # test_heat1d_chebychov(2, 3, 2, 2**5)
    # test_AdvectionDiffusion(plot=True)
    # test_heat1d_chebychov_preconditioning('D2U', True)
    # test_heat2d_chebychov(1, 1, -2, 1, 2, 'fft', 'chebychov', 2**0, 2**5)