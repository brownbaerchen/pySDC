import pytest


def get_composite_collocation_problem(L, M, N, alpha=0, dt=1e-1, restol=1e-9):
    import numpy as np
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI
    from pySDC.implementations.sweeper_classes.Q_diagonalization import QDiagonalization
    from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep, LogGlobalErrorPostRun

    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = restol

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = M
    sweeper_params['initial_guess'] = 'spread'
    sweeper_params['ignore_ic'] = True
    sweeper_params['update_f_evals'] = True

    step_params = {}
    step_params['maxiter'] = 10

    problem_params = {
        'lambdas': -1.0 * np.ones(shape=(N)),
        'u0': 1,
    }

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogLocalErrorPostStep, LogGlobalErrorPostRun]
    controller_params['mssdc_jac'] = False

    description = {}
    description['problem_class'] = testequation0d
    description['problem_params'] = problem_params
    description['sweeper_class'] = QDiagonalization
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }
    controller = controller_ParaDiag_nonMPI(**controller_args, num_procs=L, alpha=alpha)
    P = controller.MS[0].levels[0].prob

    return controller, P, description


@pytest.mark.base
@pytest.mark.parametrize('L', [1, 4])
@pytest.mark.parametrize('M', [2, 3])
@pytest.mark.parametrize('N', [1, 2])
@pytest.mark.parametrize('alpha', [1e-4, 1e-2])
def test_ParaDiag_convergence(L, M, N, alpha):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.implementations.sweeper_classes.Q_diagonalization import QDiagonalization
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.helpers.stats_helper import get_sorted

    controller, prob, description = get_composite_collocation_problem(L, M, N, alpha)
    level = controller.MS[0].levels[0]

    # setup initial conditions
    u0 = prob.u_init
    u0[:] = 1

    uend, stats = controller.run(u0=u0, t0=0, Tend=L * level.dt * 2)

    # make some tests
    error = get_sorted(stats, type='e_local_post_step')
    k = get_sorted(stats, type='niter')
    assert max(me[1] for me in k) < 9, 'ParaDiag did not converge'
    assert max(me[1] for me in error) < 1e-5, 'Error with ParaDiag too large'


@pytest.mark.base
@pytest.mark.parametrize('L', [4])
@pytest.mark.parametrize('M', [2, 3])
@pytest.mark.parametrize('N', [1])
@pytest.mark.parametrize('alpha', [1e-4, 1e-2])
def test_ParaDiag_order(L, M, N, alpha):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.implementations.sweeper_classes.Q_diagonalization import QDiagonalization
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.helpers.stats_helper import get_sorted

    errors = []
    if M == 3:
        dts = [2 ** (-x) for x in range(4, 8)]
    elif M == 2:
        dts = [2 ** (-x) for x in range(5, 9)]
    else:
        raise NotImplementedError
    Tend = max(dts) * L * 2

    for dt in dts:
        controller, prob, description = get_composite_collocation_problem(L, M, N, alpha, dt=dt)
        level = controller.MS[0].levels[0]

        # setup initial conditions
        u0 = prob.u_init
        u0[:] = 1

        uend, stats = controller.run(u0=u0, t0=0, Tend=Tend)

        # make some tests
        errors.append(get_sorted(stats, type='e_global_post_run')[-1][1])

        expected_order = level.sweep.coll.order

    errors = np.array(errors)
    dts = np.array(dts)
    order = np.log(abs(errors[1:] - errors[:-1])) / np.log(abs(dts[1:] - dts[:-1]))
    num_order = np.mean(order)

    assert (
        expected_order + 1 > num_order > expected_order
    ), f'Got unexpected numerical order {num_order} instead of {expected_order} in ParaDiag'


if __name__ == '__main__':
    test_ParaDiag_order(3, 3, 1, 1e-4)
