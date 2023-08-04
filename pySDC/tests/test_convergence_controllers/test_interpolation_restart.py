import pytest


def single_run_quench(use_interpolation):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.Quench import Quench
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
    from pySDC.implementations.convergence_controller_classes.interpolation_restarting import InterpolationRestarting
    from pySDC.implementations.hooks.log_errors import (
        LogLocalErrorPostStep,
        LogGlobalErrorPostRun,
        LogGlobalErrorPostStep,
    )
    from pySDC.implementations.hooks.log_work import LogWork

    level_params = {}
    level_params['restol'] = -1
    level_params['dt'] = 5e-2

    step_params = {}
    step_params['maxiter'] = 4

    sweeper_params = {}
    sweeper_params['num_nodes'] = 3
    sweeper_params['quad_type'] = 'RADAU-RIGHT'

    problem_params = {}

    convergence_controllers = {}
    convergence_controllers[Adaptivity] = {'e_tol': 1e-5}
    if use_interpolation:
        convergence_controllers[InterpolationRestarting.get_implementation(useMPI=False)] = {'gamma': 5e-1}

    description = {}
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['sweeper_params'] = sweeper_params
    description['sweeper_class'] = generic_implicit
    description['problem_params'] = problem_params
    description['problem_class'] = Quench
    description['convergence_controllers'] = convergence_controllers

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['mssdc_jac'] = False
    controller_params['hook_class'] = [LogLocalErrorPostStep, LogGlobalErrorPostRun, LogGlobalErrorPostStep, LogWork]

    controller = controller_nonMPI(1, controller_params, description)

    prob = controller.MS[0].levels[0].prob
    u0 = prob.u_exact(0)

    uend, stats = controller.run(u0=u0, t0=0, Tend=5e2)

    from pySDC.helpers.stats_helper import get_sorted

    e_loc = get_sorted(stats, type='e_local_post_step', recomputed=False)
    e_glob = get_sorted(stats, type='e_global_post_step', recomputed=False)
    work = get_sorted(stats, type='work_newton', recomputed=None)
    restarts = get_sorted(stats, type='restart')

    return e_loc, e_glob, work, restarts


def single_run(use_interpolation):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
    from pySDC.implementations.convergence_controller_classes.interpolation_restarting import InterpolationRestarting
    from pySDC.implementations.hooks.log_errors import (
        LogLocalErrorPostStep,
        LogGlobalErrorPostRun,
        LogGlobalErrorPostStep,
    )
    from pySDC.implementations.hooks.log_work import LogWork

    level_params = {}
    level_params['restol'] = -1
    level_params['dt'] = 5e-2

    step_params = {}
    step_params['maxiter'] = 4

    sweeper_params = {}
    sweeper_params['num_nodes'] = 3
    sweeper_params['quad_type'] = 'RADAU-RIGHT'

    problem_params = {}
    # problem_params['lambdas'] = [[-0.-1e-1j, -0-10j]]
    # problem_params['u0'] = 1.
    problem_params['mu'] = 5

    convergence_controllers = {}
    convergence_controllers[Adaptivity] = {'e_tol': 1e-5}
    if use_interpolation:
        convergence_controllers[InterpolationRestarting.get_implementation(useMPI=False)] = {'gamma': 5e-1}

    description = {}
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['sweeper_params'] = sweeper_params
    description['sweeper_class'] = generic_implicit
    description['problem_params'] = problem_params
    description['problem_class'] = vanderpol  # testequation0d
    description['convergence_controllers'] = convergence_controllers

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['mssdc_jac'] = False
    controller_params['hook_class'] = [LogLocalErrorPostStep, LogGlobalErrorPostRun, LogGlobalErrorPostStep, LogWork]

    controller = controller_nonMPI(1, controller_params, description)

    prob = controller.MS[0].levels[0].prob
    u0 = prob.u_exact(0)

    uend, stats = controller.run(u0=u0, t0=0, Tend=6e0)

    from pySDC.helpers.stats_helper import get_sorted

    e_loc = get_sorted(stats, type='e_local_post_step', recomputed=False)
    e_glob = get_sorted(stats, type='e_global_post_step', recomputed=False)
    work = get_sorted(stats, type='work_newton', recomputed=None)
    restarts = get_sorted(stats, type='restart')

    return e_loc, e_glob, work, restarts


def test_interpolation_restart_integration():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, sharex=True)

    for use_interpolation, label in zip([False, True], ['restart', 'interpolate']):
        e_loc, e_glob, work, restarts = single_run_quench(use_interpolation)
        print(e_glob)
        total_work = sum([me[1] for me in work])
        axs[0].plot([me[0] for me in e_loc], [me[1] for me in e_loc], label=f'{label}: {total_work} Newton iterations')
        axs[1].plot([me[0] for me in e_glob], [me[1] for me in e_glob])
        if not use_interpolation:
            for me in restarts:
                if me[1]:
                    axs[0].axvline(me[0], alpha=0.6)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].legend(frameon=False)
    axs[0].set_ylabel('local error')
    axs[1].set_ylabel('global error')
    axs[1].set_xlabel('time')
    plt.show()


if __name__ == '__main__':
    test_interpolation_restart_integration()
