import numpy as np
import sys
from pySDC.helpers.stats_helper import get_sorted
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


def get_description(mu=1, mode='ParaDiag'):
    level_params = {}
    level_params['dt'] = 1e-2  # / mu
    level_params['restol'] = 1e-6

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['initial_guess'] = 'copy'

    if 'Diag' in mode:
        from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization as sweeper_class

        # we only want to use the averaged Jacobian and do only one Newton iteration per ParaDiag iteration!
        newton_maxiter = 1
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        newton_maxiter = 99
        # need diagonal preconditioner for same concurrency as ParaDiag
        sweeper_params['QI'] = 'MIN-SR-S'

    from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol as problem_class

    # need to not raise an error when Newton has not converged because we do only one iteration
    problem_params = {
        'newton_maxiter': newton_maxiter,
        'crash_at_maxiter': False,
        'mu': mu,
        'newton_tol': 1e-9,
        'relative_tolerance': True,
        'stop_at_nan': False,
    }

    step_params = {}
    step_params['maxiter'] = 99

    description = {}
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description


def get_controller_params(mode='ParaDiag'):
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_work import LogWork, LogSDCIterations

    controller_params = {}
    controller_params['logger_level'] = 100
    controller_params['hook_class'] = [LogGlobalErrorPostRun, LogWork, LogSDCIterations]

    if 'Diag' in mode:
        controller_params['alpha'] = 1e-8

        # For nonlinear problems, we need to communicate the average solution, which allows to compute the average
        # Jacobian locally. For linear problems, we do not want the extra communication.
        controller_params['average_jacobians'] = True
    else:
        # We do Block-Jacobi multi-step SDC here. It's a bit silly but it's better for comparing "speedup"
        controller_params['mssdc_jac'] = True

    return controller_params


def run_vdp(
    mu=1,
    n_steps=4,
    mode='ParaDiag',
):
    if 'Diag' in mode:
        from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import (
            controller_ParaDiag_nonMPI as controller_class,
        )
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI as controller_class

    if mode == 'serial':
        num_procs = 1
    else:
        num_procs = n_steps

    description = get_description(mu, mode)
    controller_params = get_controller_params(mode)

    if mode == 'PFASSTDiag':
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

        description_PFASST = get_description(mu=mu, mode='PFASST')
        description_PFASST['step_params']['maxiter'] = 1
        controller_params_PFASST = get_controller_params('PFASST')
        controller_params_PFASST['logger_level'] = 30

        controller_PFASST = controller_nonMPI(
            num_procs=num_procs, description=description_PFASST, controller_params=controller_params_PFASST
        )
        controller_params['PFASST_controller'] = controller_PFASST

        for S in controller_PFASST.MS:
            S.levels[0].prob.init = tuple([*S.levels[0].prob.init[:2]] + [np.dtype('complex128')])

    controller = controller_class(num_procs=num_procs, description=description, controller_params=controller_params)

    for S in controller.MS:
        S.levels[0].prob.init = tuple([*S.levels[0].prob.init[:2]] + [np.dtype('complex128')])

    P = controller.MS[0].levels[0].prob

    t0 = 0.0
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=n_steps * controller.MS[0].levels[0].dt)

    k_Newton = get_sorted(stats, type='work_newton')

    if mode == 'PFASSTDiag':
        k_Newton_PFASST = get_sorted(controller.params.PFASST_controller.return_stats(), type='work_newton')
        assert len(k_Newton) == len(k_Newton_PFASST)
        _k_Newton_tot = sum(me[1] for me in k_Newton)
        k_Newton = [(k_Newton[i][0], k_Newton[i][1] + k_Newton_PFASST[i][1]) for i in range(len(k_Newton))]
        if sum(me[1] for me in k_Newton) > _k_Newton_tot:
            print('Warning: Did not record any Newton iterations in PFASST!')

    if mode == 'serial':
        k_Newton_per_step = sum(me[1] for me in k_Newton)
    else:
        k_Newton_per_step = max(me[1] for me in k_Newton)

    # check if the method has converged
    k = get_sorted(stats, type='niter')
    converged = max(me[1] for me in k) < description['step_params']['maxiter']

    error = get_sorted(stats, type='e_global_post_run')[-1][1]

    return uend, k_Newton_per_step, converged, error


def get_iteration_counts(mode, mu_range, steps_range):
    newton_iter = np.zeros((len(mu_range), len(steps_range)))
    errors = np.zeros_like(newton_iter)

    for i, mu in enumerate(mu_range):

        for j, n_steps in enumerate(steps_range):
            sol, k_Newton, converged, error = run_vdp(mu=mu, n_steps=n_steps, mode=mode)

            errors[i, j] = error
            if converged:
                newton_iter[i, j] = k_Newton
            else:
                break

            print(f'{mode} with {mu=:4f} and {n_steps=:4f} needed {k_Newton:6f} Newton iterations')

    with open(f'data/iteration_matrix_vdp_{mode}.pickle', 'wb') as file:
        data = {'newton_iter': newton_iter, 'mu': mu_range, 'n_steps': steps_range, 'errors': errors}
        pickle.dump(data, file)


def plot_iteration_counts():
    modes = ['serial', 'ParaDiag', 'PFASST', 'PFASSTDiag']
    fig, axs = plt.subplots(1, len(modes), figsize=(len(modes) * 3, 3.5), sharex=True, sharey=True)
    plt.rcParams['figure.constrained_layout.use'] = True
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes('right', size='3%', pad=0.03)

    all_data = {}
    for mode in modes:
        with open(f'data/iteration_matrix_vdp_{mode}.pickle', 'rb') as file:
            all_data[mode] = pickle.load(file)

    maximum = max(np.max(all_data[mode]['newton_iter']) for mode in modes)
    plt.rcParams['hatch.color'] = 'gold'

    for i, mode in enumerate(modes):
        data = all_data[mode]

        mu, n_steps = np.meshgrid(data['mu'], data['n_steps'], indexing='ij')
        mask = data['newton_iter'] == 0
        _plot = data['newton_iter'].copy()
        _plot[mask] = np.nan

        im = axs[i].pcolormesh(
            mu, n_steps, np.log10(_plot), vmin=1, vmax=np.log10(maximum), cmap='viridis', rasterized=True
        )

        # find out where this method is best
        best = data['newton_iter'] > 0
        for _mode in modes:
            other_not_converged = all_data[_mode]['newton_iter'] == 0
            better = np.logical_or(data['newton_iter'] <= all_data[_mode]['newton_iter'], other_not_converged)
            best = np.logical_and(best, better)

        axs[i].contourf(mu, n_steps, best, hatches=['', '*'], colors='none')

        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_title(mode)
        axs[i].set_xlabel(r'$\mu$')
        axs[i].set_aspect(1)
        axs[i].set_xlim((min(data['mu']), max(data['mu'])))
        axs[i].set_ylim((min(data['n_steps']), max(data['n_steps'])))

    axs[0].set_ylabel('number of parallel steps')
    fig.colorbar(im, cax, label=r'$\log_{10}(\text{# Newton iterations per step})$')
    fig.tight_layout()
    fig.savefig('compare_parallelization_schemes_van_der_pol.pdf')
    plt.show()


if __name__ == '__main__':

    mu_range = np.linspace(0, 20, 21)
    steps_range = np.arange(12) + 1  # np.linspace(1, 13, 12)#2**np.arange(7)

    mu_range = 2 ** np.arange(7)
    steps_range = mu_range.copy()
    # for mode in ['serial', 'PFASST', 'ParaDiag', 'PFASSTDiag']:
    #     get_iteration_counts(mode, mu_range, steps_range)
    plot_iteration_counts()
