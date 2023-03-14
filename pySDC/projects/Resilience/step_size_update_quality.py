import matplotlib.pyplot as plt
import numpy as np
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate
from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityCollocation

from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.Schroedinger import run_Schroedinger
from pySDC.projects.Resilience.Lorenz import run_Lorenz
from pySDC.projects.Resilience.piline import run_piline


class AlwaysRestart(LogEmbeddedErrorEstimate, LogLocalErrorPostStep):
    def __init__(self):
        super().__init__()
        self.restarted = False
        self.log_now = False

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        if not self.restarted and step.status.first:
            step.status.restart = True
            self.restarted = True
            self.log_now = True
        else:
            if step.status.last and self.log_now:
                self.log_error(step, level_number, appendix='_post_restart')
                self.log_local_error(step, level_number, '_post_restart')
                self.log_now = False
            elif step.status.first:
                self.restarted = False

            step.status.restart = False


def get_e_post_restart(problem, description, num_procs=1, Tend=None, **kwargs):
    stats, controller, _ = problem(
        custom_description=description,
        hook_class=[AlwaysRestart],
        custom_controller_params={'logger_level': 30},
        num_procs=num_procs,
        Tend=Tend,
    )
    e_loc = get_sorted(stats, type='e_local_post_restart')
    if AdaptivityCollocation in description['convergence_controllers'].keys():
        return get_sorted(stats, type='error_embedded_estimate_collocation_post_restart'), e_loc
    else:
        return get_sorted(stats, type='error_embedded_estimate_post_restart'), e_loc


def get_description(adaptivity_convergence_controller, e_tol, QI='IE', use_semi_global_estimate=False, **kwargs):
    convergence_controllers = {}
    convergence_controllers[adaptivity_convergence_controller] = {
        'e_tol': e_tol,
        'beta': 1.0,
        'use_semi_global_estimate': use_semi_global_estimate,
        'adaptive_coll_params': {'quad_type': ['RADAU-RIGHT', 'GAUSS']},
    }

    step_params = {}
    level_params = {}

    sweeper_params = {}
    sweeper_params['QI'] = QI

    if adaptivity_convergence_controller == AdaptivityCollocation:
        step_params['maxiter'] = 99
        level_params['restol'] = e_tol / 10
    elif adaptivity_convergence_controller == Adaptivity:
        step_params['maxiter'] = 5
        level_params['restol'] = -1

    description = {}
    description['convergence_controllers'] = convergence_controllers
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params
    description['level_params'] = level_params

    return description


def plot_e_post_restart(label='', ax=None, color='blue', **kwargs):
    e_post_restart, e_loc = get_e_post_restart(description=get_description(**kwargs), **kwargs)

    violation = np.mean([abs(me[1] - kwargs['e_tol']) for me in e_post_restart][2:-2])
    print(f'{label}: Delta={violation:.2e}, n={len(e_post_restart)}')

    ax.plot(
        [me[0] for me in e_post_restart][2:-2],
        [me[1] for me in e_post_restart][2:-2],
        label=fr'{label}, $\bar{{\Delta}}={{{violation:.2e}}}$',
        color=color,
    )
    ax.plot(
        [me[0] for me in e_loc][2:-2], [me[1] for me in e_loc][2:-2], label=fr'{label}, actual', color=color, ls='--'
    )

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\epsilon_\mathrm{local}$')


def plot_single(ax, problem, e_tol=1e-7, num_procs=4):
    titles = {
        run_vdp: 'van der Pol',
        run_advection: 'advection',
        run_Schroedinger: 'Schr\"odinger',
        run_Lorenz: 'Lorenz attractor',
        run_piline: 'Pi-line',
    }

    ax.axhline(e_tol, color='black', label=r'$\epsilon_\mathrm{tol}$')
    plot_e_post_restart(
        problem=problem,
        adaptivity_convergence_controller=Adaptivity,
        e_tol=e_tol,
        num_procs=1,
        QI='IE',
        label='serial',
        ax=ax,
        use_semi_global_estimate=False,
        Tend=10.0,
        color='green',
        ls=':',
    )
    for error in [True, False]:
        label = 'semi-global' if error else 'linearised local'
        color = 'blue' if error else 'red'
        plot_e_post_restart(
            problem=problem,
            adaptivity_convergence_controller=Adaptivity,
            e_tol=e_tol,
            num_procs=num_procs,
            QI='IE',
            label=f'{label}',
            ax=ax,
            use_semi_global_estimate=error,
            Tend=10.0,
            color=color,
        )

    ax.set_title(titles[problem])
    ax.set_yscale('log')
    ax.legend(frameon=False)


def plot_all(num_procs):
    setup_mpl(font_size=5, reset=True)

    e_tol = 1e-7

    fig, axs = plt.subplots(
        2, 2, figsize=figsize_by_journal('Springer_Numerical_Algorithms', 1.0, 1.0), sharex=True, sharey=True
    )

    problems = [run_advection, run_Lorenz, run_piline, run_vdp]
    for i in range(len(problems)):
        plot_single(axs.flatten()[i], problems[i], e_tol, num_procs=num_procs)

    for ax in [axs[0, 0], axs[0, 1], axs[1, 1]]:
        ax.set_ylabel('')
        ax.set_xlabel('')

    fig.tight_layout()
    fig.savefig(f'data/step_size_update-{num_procs}procs.pdf')


if __name__ == '__main__':
    plot_all(4)
    plt.show()
