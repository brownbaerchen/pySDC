import pytest


def run_simulation(sweeper_class, nsteps, nsweeps, nnodes):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_work import LogWork

    from pySDC.projects.RayleighBenard.tests.test_RBC_3D_analysis import get_args
    from pySDC.projects.RayleighBenard.RBC3D_configs import get_config

    args = get_args('.')
    config = get_config(args)

    description = config.get_description(res=8)
    description['level_params']['nsweeps'] = nsweeps
    description['sweeper_params']['num_nodes'] = nnodes

    controller_params = config.get_controller_params()
    controller_params['hook_class'] = [LogWork]

    controller = controller_nonMPI(1, controller_params, description)

    u0 = controller.MS[0].levels[0].prob.u_exact(0)
    dt = description['level_params']['dt']

    return controller.run(u0, 0, nsteps * dt)


@pytest.mark.parametrize('nsweeps', [1, 2, 3])
@pytest.mark.parametrize('nnodes', [1, 2, 3])
def test_serial_optimized_sweeper(nsweeps, nnodes):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.projects.RayleighBenard.sweepers import imex_1st_order_diagonal_serial

    args = {
        'nsteps': 4,
        'nsweeps': nsweeps,
        'nnodes': nnodes,
    }

    u_opt, stats_opt = run_simulation(imex_1st_order_diagonal_serial, **args)
    u_normal, stats_normal = run_simulation(imex_1st_order, **args)

    assert np.allclose(u_normal, u_opt), 'Got different result with optimized sweeper'

    expect_work = (nsweeps - 1) * nnodes + 1
    got_rhs = [me[1] for me in get_sorted(stats_opt, type='work_rhs')]
    assert np.allclose(expect_work, got_rhs), f'Expected {expect_work} right hand side evaluations, but did {got_rhs}'

    got_solves = [me[1] for me in get_sorted(stats_opt, type='work_cached_direct')]
    assert np.allclose(expect_work, got_solves), f'Expected {expect_work} solves, but did {got_solves}'


if __name__ == '__main__':
    test_serial_optimized_sweeper(2, 2)
