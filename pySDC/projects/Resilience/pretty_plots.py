# script to make plots for the paper
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import setup_mpl
import matplotlib as mpl

from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

cm = 1 / 2.54


def savefig(fig, name):
    """
    Save a figure to some predefined location.

    Args:
        fig (Matplotlib.Figure): The figure of the plot
        name (str): The name of the plot

    Returns:
        None
    """
    fig.tight_layout()
    path = f'data/paper/{name}.pdf'
    fig.savefig(path, bbox_inches='tight', transparent=True)
    print(f'saved "{path}"')


def plot_phase_space_things():
    """
    Make a phase space plots comparing van der Pol with and without adaptivity
    """

    def plot_phase_space(stats, ax, color, marker, label, rescale=1.0):
        """
        Plot the solution over time in phase space

        Args:
            stats (pySDC.stats): The stats object of the run
            ax (Matplotlib.pyplot.axes): Somewhere to plot
            color, marker, label (str): Plotting attributes

        Returns:
            None
        """
        u = np.array([me[1][0] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])
        p = np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=False, sortby='time')])

        if rescale:
            fac = max(abs(np.append(u, p)))
        else:
            fac = 1.0

        ax.plot(u / fac, p / fac, color=color, marker=marker, label=label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r'$u_t$')
        ax.set_xlabel(r'$u$')

    setup_mpl(font_size=8, reset=True)
    mpl.rcParams.update({'lines.markersize': 8})
    mu_range = [0, 5, 10]
    Tend_range = [6.4, 12, 19]
    markers = ['.', 'v', '1']
    nsteps = [30, 300, 1000]
    colors = ['blue', 'orange', 'magenta']
    fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 8.8 * cm), sharex=True, sharey=True)
    convergence_controllers = {}
    titles = [r'Fixed $\Delta t$', 'Adaptivity']
    for j in range(2):
        ax = axs[j]
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        ax.set_title(titles[j])

        if j > 0:
            convergence_controllers[Adaptivity] = {'e_tol': 3e-5}
            print('Activating adaptivity')

        for i in range(len(mu_range)):
            problem_params = {'mu': mu_range[i]}
            custom_description = {
                'level_params': {'dt': Tend_range[i] / nsteps[i]},
                'convergence_controllers': convergence_controllers,
            }
            stats, controller, Tend = run_vdp(
                custom_description=custom_description, custom_problem_params=problem_params, Tend=Tend_range[i]
            )
            num_steps = len(get_sorted(stats, type='u')) - 1
            plot_phase_space(
                stats,
                ax,
                color=colors[i],
                label=rf'$\mu={{{mu_range[i]}}}, N={num_steps}$',
                marker=markers[i],
                rescale=True,
            )

            # check resolution
            u = np.array([me[1] for me in get_sorted(stats, type='u')])
            increment = np.linalg.norm(u[1:] - u[:-1], axis=1)

            print(
                f'Mu={mu_range[i]:2d}, phase space dist: max={increment.max():.2e}, mean={increment.mean():.2e}, min={increment.min():.2e}, std={increment.mean():.2e}'
            )
        lim = max(np.append(ax.get_ylim(), ax.get_xlim()))
        ax.set_ylim([-lim, lim])
        ax.set_xlim([-lim, lim])
        ax.legend(frameon=True, loc='lower right')

        if j > 0:
            ax.set_xlabel('')
            ax.set_ylabel('')
    fig.tight_layout()
    savefig(fig, 'vdp_phase_space')


def plot_adaptivity_stuff():
    # TODO: docs
    from pySDC.projects.Resilience.hook import log_global_error, log_local_error

    setup_mpl(font_size=8, reset=True)
    mpl.rcParams.update({'lines.markersize': 8})
    markers = ['.', 'v', '1']
    nsteps = [30, 300, 1000]
    fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 8.8 * cm), sharex=True, sharey=False)
    convergence_controllers = {}
    ax = axs[0]
    labels = [r'Fixed $\Delta t$', 'Adaptivity']

    def plot_error(stats, ax, iter_ax, **kwargs):
        """
        Plot global error and cumulative sum of iterations

        Args:
            stats (dict): Stats from pySDC run
            ax (Matplotlib.pyplot.axes): Somewhere to plot the error
            iter_ax (Matplotlib.pyplot.axes): Somewhere to plot the iterations

        Returns:
            None
        """
        e = get_sorted(stats, type='e_loc', recomputed=False)
        ax.plot([me[0] for me in e], [me[1] for me in e], ls='-', **kwargs)
        k = get_sorted(stats, type='sweeps')
        iter_ax.plot([me[0] for me in k], np.cumsum([me[1] for me in k]), ls='-')
        ax.set_yscale('log')
        ax.set_ylabel(r'$e_\mathrm{loc}$')
        iter_ax.set_ylabel(r'$k$')
        ax.set_xlabel(r'$t$')

    for i in range(2):
        if i > 0:
            convergence_controllers[Adaptivity] = {'e_tol': 1e-6}
            print('Activating adaptivity')
        problem_params = {'mu': 5.0}
        custom_description = {
            'level_params': {'dt': 2e-2},
            'convergence_controllers': convergence_controllers,
        }
        stats, controller, Tend = run_vdp(
            custom_description=custom_description,
            custom_problem_params=problem_params,
            Tend=10.0,
            hook_class=log_local_error,
        )
        plot_error(stats, axs[0], axs[1], label=labels[i])

    ax.legend(frameon=True, loc='lower right')

    fig.tight_layout()
    savefig(fig, 'adaptivity')


def plot_recovery_rate():
    """
    Make plots showing the recovery rate for all strategies under consideration.
    """
    from pySDC.projects.Resilience.fault_stats import (
        FaultStats,
        BaseStrategy,
        AdaptivityStrategy,
        IterateStrategy,
        HotRodStrategy,
    )

    setup_mpl(reset=True, font_size=8)
    mpl.rcParams.update({'lines.markersize': 8})

    stats_analyser = FaultStats(
        prob=run_vdp,
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy(), HotRodStrategy()],
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        num_procs=1,
        mode='combination',
    )
    stats_analyser.run_stats_generation(runs=5000, step=50)
    mask = None

    fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 7 * cm), sharex=True, sharey=True)
    stats_analyser.plot_things_per_things(
        'recovered', 'bit', False, op=stats_analyser.rec_rate, mask=mask, args={'ylabel': 'recovery rate'}, ax=axs[0]
    )
    not_crashed = None
    for i in range(len(stats_analyser.strategies)):
        not_crashed = stats_analyser.get_mask(strategy=stats_analyser.strategies[i], key='error', op='uneq', val=np.inf)
        fixable = stats_analyser.get_mask(key='node', op='gt', val=0, old_mask=not_crashed)

        if type(stats_analyser.strategies[i]) == AdaptivityStrategy:
            fixable = stats_analyser.get_mask(key='iteration', op='lt', val=3, old_mask=fixable)

        stats_analyser.plot_things_per_things(
            'recovered',
            'bit',
            False,
            op=stats_analyser.rec_rate,
            mask=fixable,
            args={'ylabel': '', 'xlabel': ''},
            ax=axs[1],
            fig=fig,
            strategies=[stats_analyser.strategies[i]],
        )
    axs[1].get_legend().remove()
    axs[0].set_title('All faults')
    axs[1].set_title('Only recoverable faults')
    fig.tight_layout()
    savefig(fig, 'recovery_rate_compared')


def plot_fault():
    from pySDC.projects.Resilience.fault_stats import (
        FaultStats,
        BaseStrategy,
        log_local_error,
    )

    stats_analyser = FaultStats(
        prob=run_vdp,
        strategies=[BaseStrategy()],
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        num_procs=1,
        mode='combination',
    )

    # run = 779  # 120, 11, 780

    fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 7 * cm), sharex=True, sharey=True)
    colors = ['blue', 'red']
    ls = ['--', '--', '-', '-']
    markers = ['*', '.']
    do_faults = [False, False, True, True]
    labels = ['*', '*', '', '']
    # runs = [779, 120, 779, 120]
    runs = [779, 810, 779, 923]
    for i in range(len(do_faults)):
        ax = axs[i % 2]
        stats, controller, Tend = stats_analyser.single_run(
            strategy=BaseStrategy(), run=runs[i], faults=do_faults[i], hook_class=log_local_error
        )
        u = get_sorted(stats, type='u')
        faults = get_sorted(stats, type='bitflip')
        ax.plot(
            [me[0] for me in u],
            [me[1][0] for me in u],
            ls=ls[i],
            color=colors[0],
            label=rf'$u{{{labels[i]}}}$',
            marker=markers[0],
            markevery=15,
        )
        ax.plot(
            [me[0] for me in u],
            [me[1][1] for me in u],
            ls=ls[i],
            color=colors[1],
            label=rf'$u^{{{labels[i]}}}_t$',
            marker=markers[1],
            markevery=15,
        )
        for idx in range(len(faults)):
            ax.axvline(faults[idx][0], color='black', label='Fault', ls=':')
            print(
                f'Fault at t={faults[idx][0]:.2e}, iter={faults[idx][1][1]}, node={faults[idx][1][2]}, space={faults[idx][1][3]}, bit={faults[idx][1][4]}'
            )
            ax.set_title(f'Fault in bit {faults[idx][1][4]}')

    axs[0].legend(frameon=False)
    axs[0].set_xlabel(r'$t$')
    savefig(fig, 'fault')


if __name__ == "__main__":
    plot_adaptivity_stuff()
    # plot_fault()
    # plot_recovery_rate()
    # plot_phase_space_things()
