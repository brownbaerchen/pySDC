# script to make plots for the paper
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import setup_mpl
import matplotlib as mpl

from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity


def savefig(fig, name):
    """
    Save a figure to some predefined location.

    Args:
        fig (Matplotlib.Figure): The figure of the plot
        name (str): The name of the plot

    Returns:
        None
    """
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

    setup_mpl(font_size=12, reset=True)
    mpl.rcParams.update({'lines.markersize': 8})
    mu_range = [0, 5, 10]
    Tend_range = [6.5, 12, 19]
    markers = ['.', 'v', '1']
    nsteps = [28, 300, 1000]
    colors = ['blue', 'orange', 'magenta']
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.7), sharex=True, sharey=True)
    convergence_controllers = {}
    titles = [r'Fixed $\Delta t$', 'Adaptivity']
    for j in range(2):
        ax = axs[j]
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        ax.set_title(titles[j])

        if j > 0:
            convergence_controllers[Adaptivity] = {'e_tol': 5e-5}
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
        ax.legend(frameon=False, loc='lower right')

        if j > 0:
            ax.set_xlabel('')
            ax.set_ylabel('')
    fig.tight_layout()
    savefig(fig, 'vdp_phase_space')


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

    setup_mpl(reset=True, font_size=12)
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

    fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)
    stats_analyser.plot_things_per_things(
        'recovered', 'bit', False, op=stats_analyser.rec_rate, mask=mask, args={'ylabel': 'recovery rate'}, ax=axs[0]
    )
    not_crashed = None
    for i in range(len(stats_analyser.strategies)):
        not_crashed = stats_analyser.get_mask(
            strategy=stats_analyser.strategies[i], key='error', op='uneq', val=np.inf, old_mask=not_crashed
        )
    fixable = stats_analyser.get_mask(key='node', op='gt', val=0, old_mask=not_crashed)
    stats_analyser.plot_things_per_things(
        'recovered',
        'bit',
        False,
        op=stats_analyser.rec_rate,
        mask=fixable,
        args={'ylabel': '', 'xlabel': ''},
        ax=axs[1],
        fig=fig,
    )
    axs[1].get_legend().remove()
    axs[0].set_title('All faults')
    axs[1].set_title('Only recoverable faults')
    fig.tight_layout()
    savefig(fig, 'recovery_rate_compared')


if __name__ == "__main__":
    plot_recovery_rate()
    plot_phase_space_things()
    plt.show()
