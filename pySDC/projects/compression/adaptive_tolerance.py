import numpy as np
from pySDC.projects.compression.compression_convergence_controller import AdaptiveCompression
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun


def single_run(problem, e_tol, compression_ratio):
    adaptiviy_params = {}
    adaptiviy_params['e_tol'] = e_tol

    description = {}
    description['convergence_controllers'] = {
        AdaptiveCompression: {'adaptivity_args': adaptiviy_params, 'compression_ratio': compression_ratio},
    }

    controller_params = {'logger_level': 30}

    stats, _, _ = problem(
        custom_description=description, custom_controller_params=controller_params, hook_class=LogGlobalErrorPostRun
    )
    e_global = [me[1] for me in get_sorted(stats, type='e_global_post_run')]
    if len(e_global) == 0:
        return np.inf
    else:
        return min(e_global)


def multiple_runs(problem, e_tol, compression_ratio_range):
    # do a reference run
    e_star = single_run(problem, e_tol, 0)

    # do the other runs
    e = []
    for me in compression_ratio_range:
        e += [single_run(problem, e_tol, me) / e_star]

    return compression_ratio_range, e


def multiple_multiple_runs(problem, e_tol_range, compression_ratio_range):
    es = []
    for me in e_tol_range:
        es += [multiple_runs(problem, me, compression_ratio_range)[1]]
    return compression_ratio_range, e_tol_range, es


def plot(problem, e_tol_range, compression_ratio_range, fig_name):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))

    r, eps, e = multiple_multiple_runs(problem, e_tol_range, compression_ratio_range)

    for i in range(len(eps)):
        ax.loglog(r, e[i], label=fr'$\epsilon={{{eps[i]:.2e}}}$')

    ax.legend(frameon=False)
    ax.set_xlabel(r'$\frac{\delta}{\epsilon \Delta t}$')
    ax.set_ylabel(r'$\frac{e}{e^*}$')

    fig.tight_layout()
    fig.savefig(f'adaptive_nonsense_{fig_name}.pdf')


def advection_plot():
    from pySDC.projects.Resilience.advection import run_advection

    compression_ratio_range = np.logspace(-4, 1, 10)
    e_tol_range = np.logspace(-7, -4, 4)
    plot(run_advection, e_tol_range, compression_ratio_range, 'advection')


def vdp_plot():
    from pySDC.projects.Resilience.vdp import run_vdp

    compression_ratio_range = np.logspace(-2, 3, 12)
    e_tol_range = np.logspace(-7, -4, 4)
    plot(run_vdp, e_tol_range, compression_ratio_range, 'vdp')


if __name__ == '__main__':
    advection_plot()
