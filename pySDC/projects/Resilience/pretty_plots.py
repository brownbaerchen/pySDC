# script to make plots for the paper
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import setup_mpl
import matplotlib as mpl

from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.Lorenz import run_Lorenz
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


def get_vdp_fault_stats(mode='paper'):
    """
    Retrieve fault statistics for van der Pol equation.
    """
    from pySDC.projects.Resilience.fault_stats import (
        FaultStats,
        BaseStrategy,
        AdaptivityStrategy,
        IterateStrategy,
        HotRodStrategy,
    )

    if mode == 'paper':
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy(), HotRodStrategy()]
    elif mode == 'talk_CSE23':
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy()]

    strategies = [HotRodStrategy()]
    stats_analyser = FaultStats(
        prob=run_Lorenz,
        strategies=strategies,
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        num_procs=1,
        mode='combination',
    )
    stats_analyser.run_stats_generation(runs=5000, step=50)
    stats_analyser.get_recovered()
    print(len(stats_analyser.load()['iteration']))
    return stats_analyser


def plot_efficiency():
    # TODO: docs

    setup_mpl(reset=True, font_size=8)
    mpl.rcParams.update({'lines.markersize': 8})

    stats_analyser = get_vdp_fault_stats()

    mask = stats_analyser.get_mask()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8 * cm, 8 * cm))

    res = {}
    for strategy in stats_analyser.strategies:
        dat = stats_analyser.load(strategy=strategy, faults=True)
        dat_no_faults = stats_analyser.load(strategy=strategy, faults=False)

        fail_rate = 1.0 - stats_analyser.rec_rate(dat, dat_no_faults, 'recovered', mask)
        iterations_no_faults = np.mean(dat_no_faults['total_iteration'])

        detected = stats_analyser.get_mask(strategy=strategy, key='total_iteration', op='gt', val=iterations_no_faults)
        rec_mask = stats_analyser.get_mask(strategy=strategy, key='recovered', op='eq', val=True, old_mask=detected)
        if rec_mask.any():
            extra_iterations = np.mean(dat['total_iteration'][rec_mask]) - iterations_no_faults
        else:
            extra_iterations = 0

        res[strategy.name] = [fail_rate, extra_iterations, iterations_no_faults]

    # normalize
    # for strategy in stats_analyser.strategies:
    norms = [max([res[k][i] for k in res.keys()]) for i in range(len(res['base']))]
    res_norm = res.copy()
    for k in res_norm.keys():
        for i in range(3):
            res_norm[k][i] /= norms[i]

    theta = np.array([30, 150, 270, 30]) * 2 * np.pi / 360
    for s in stats_analyser.strategies:
        ax.plot(theta, res_norm[s.name] + [res_norm[s.name][0]], label=s.name, color=s.color, marker=s.marker)

    labels = ['fail rate', 'extra iterations\nfor recovery', 'iterations for solution']
    ax.set_xticks(theta[:-1], [f'{labels[i]}\nmax={norms[i]:.2f}' for i in range(len(labels))])
    ax.set_rlabel_position(90)

    ax.legend(frameon=True, loc='lower right')
    savefig(fig, 'efficiency')


def plot_vdp_solution():
    """
    Plot the solution of van der Pol problem over time to illustrate the varying time scales.
    """
    setup_mpl(font_size=8, reset=True)
    mpl.rcParams.update({'lines.markersize': 8})
    fig, ax = plt.subplots(figsize=(9 * cm, 8 * cm))

    custom_description = {'convergence_controllers': {Adaptivity: {'e_tol': 1e-7}}}
    problem_params = {}

    stats, _, _ = run_vdp(
        custom_description=custom_description, custom_problem_params=problem_params, Tend=28.6
    )

    u = get_sorted(stats, type='u')
    ax.plot([me[0] for me in u], [me[1][0] for me in u], color='black')
    ax.set_ylabel(r'$u$')
    ax.set_xlabel(r'$t$')
    savefig(fig, 'vdp_sol')


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
    savefig(fig, 'vdp_phase_space')


def plot_adaptivity_stuff():
    # TODO: docs
    from pySDC.projects.Resilience.fault_stats import AdaptivityStrategy, BaseStrategy, IterateEmbeddedStrategy
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI
    from pySDC.implementations.hooks.log_errors import LogLocalError, LogGlobalError

    stats_analyser = get_vdp_fault_stats()

    setup_mpl(font_size=8, reset=True)
    mpl.rcParams.update({'lines.markersize': 6})
    #fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 7 * cm), sharex=True, sharey=False)
    fig, axs = plt.subplots(3, 1, figsize=(10 * cm, 11 * cm), sharex=True, sharey=False)

    def plot_error(stats, ax, iter_ax, strategy, **kwargs):
        """
        Plot global error and cumulative sum of iterations

        Args:
            stats (dict): Stats from pySDC run
            ax (Matplotlib.pyplot.axes): Somewhere to plot the error
            iter_ax (Matplotlib.pyplot.axes): Somewhere to plot the iterations
            strategy (pySDC.projects.Resilience.fault_stats.Strategy): The resilience strategy

        Returns:
            None
        """
        e = get_sorted(stats, type='error_embedded_estimate', recomputed=False)
        ax.plot([me[0] for me in e], [me[1] for me in e], markevery=15, **strategy.style, **kwargs)
        k = get_sorted(stats, type='k')
        iter_ax.plot([me[0] for me in k], np.cumsum([me[1] for me in k]), **strategy.style, markevery=15, **kwargs)
        ax.set_yscale('log')
        ax.set_ylabel(r'$e_\mathrm{loc}$')
        iter_ax.set_ylabel(r'iterations')

    force_params = {'convergence_controllers': {EstimateEmbeddedErrorNonMPI: {}}}
    for strategy in [AdaptivityStrategy, BaseStrategy, IterateEmbeddedStrategy]:
        stats, _, _ = stats_analyser.single_run(strategy=strategy(), force_params=force_params)
        plot_error(stats, axs[1], axs[2], strategy())

        if strategy == AdaptivityStrategy:
            u = get_sorted(stats, type='u')
            axs[0].plot([me[0] for me in u], [me[1][0] for me in u], color='black', label=r'$u$')
            axs[0].plot([me[0] for me in u], [me[1][1] for me in u], color='black', ls='--', label=r'$u_t$')
            axs[0].legend(frameon=False)


    axs[1].set_ylim(bottom=1e-9)
    axs[2].set_xlabel(r'$t$')
    axs[0].set_ylabel('solution')
    axs[2].legend(frameon=False)
    savefig(fig, 'adaptivity')


def plot_adaptivity_stuff_2():
    # TODO: docs
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI
    from pySDC.projects.Resilience.convergence_by_embedded_error_estimate import CheckConvergenceEmbeddedErrorEstimate
    from pySDC.implementations.hooks.log_errors import LogLocalError, LogGlobalError
    from pySDC.projects.Resilience.hook import LogData

    setup_mpl(font_size=8, reset=True)
    mpl.rcParams.update({'lines.markersize': 8})
    markers = ['.', 'v', '1']
    fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 7 * cm), sharex=True, sharey=False)
    convergence_controllers = {}
    ax = axs[0]
    labels = [r'Fixed $\Delta t$', 'Adaptivity', 'Iterate']

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
        e = get_sorted(stats, type='error_embedded_estimate', recomputed=False)
        ax.plot([me[0] for me in e], [me[1] for me in e], ls='-', **kwargs)
        k = get_sorted(stats, type='k')
        iter_ax.plot([me[0] for me in k], np.cumsum([me[1] for me in k]), ls='-')
        ax.set_yscale('log')
        ax.set_ylabel(r'$e_\mathrm{loc}$')
        iter_ax.set_ylabel(r'$k$')
        ax.set_xlabel(r'$t$')

    convergence_controllers[EstimateEmbeddedErrorNonMPI] = {}
    for i in range(3):
        if i == 1:
            convergence_controllers[Adaptivity] = {'e_tol': 2e-6}
            print('Adaptivity')
        if i == 2:
            convergence_controllers.pop(Adaptivity)
            convergence_controllers[CheckConvergenceEmbeddedErrorEstimate] = {'e_tol': 2e-6}
            print('Iterate')
                     
        problem_params = {'mu': 5.0}
        custom_description = {
            'level_params': {'dt': 1e-2},
            'convergence_controllers': convergence_controllers,
        }
        stats, controller, Tend = run_vdp(
            custom_description=custom_description,
            custom_problem_params=problem_params,
            Tend=10.0,
            hook_class=[LogLocalError, LogData],
        )
        plot_error(stats, axs[0], axs[1], label=labels[i])

    ax.legend(frameon=True, loc='lower right')
    ax.set_ylim(bottom=1e-9)

    savefig(fig, 'adaptivity2')


def plot_recovery_rate_talk_CSE23():
    """
    Make plots showing the recovery rate for all strategies under consideration.
    Plot made for talk at SIAM CSE23
    """
    from pySDC.projects.Resilience.fault_stats import AdaptivityStrategy

    setup_mpl(reset=True, font_size=8)
    mpl.rcParams.update({'lines.markersize': 8})

    stats_analyser = get_vdp_fault_stats(mode='talk_CSE23')
    mask = None

    fig, ax = plt.subplots(1, 1, figsize=(9 * cm, 7 * cm))
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
            args={'ylabel': 'recovery rate'},
            ax=ax,
            fig=fig,
            strategies=[stats_analyser.strategies[i]],
        )
    savefig(fig, 'recovery_rate_CSE23')


def plot_recovery_rate():
    """
    Make plots showing the recovery rate for all strategies under consideration.
    """
    from pySDC.projects.Resilience.fault_stats import AdaptivityStrategy

    setup_mpl(reset=True, font_size=8)
    mpl.rcParams.update({'lines.markersize': 8})

    stats_analyser = get_vdp_fault_stats()
    #stats_analyser.scrutinize(AdaptivityStrategy(), 2)
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
            dat = stats_analyser.load()
            max_iter = max(dat['iteration'])
            fixable = stats_analyser.get_mask(key='iteration', op='lt', val=max_iter, old_mask=fixable)

            not_fixed = stats_analyser.get_mask(key='recovered', op='eq', val=False, old_mask=fixable)
            print(dat['recovered'][not_fixed])
            print(stats_analyser.strategies[i].name)
            #stats_analyser.print_faults(not_fixed)

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
    savefig(fig, 'recovery_rate_compared')


def plot_fault():
    from pySDC.projects.Resilience.fault_stats import (
        FaultStats,
        BaseStrategy,
        LogLocalError,
    )
    from pySDC.projects.Resilience.hook import LogData

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
    colors = ['blue', 'red', 'magenta']
    ls = ['--', '--', '-', '-']
    markers = ['*', '.', 'y']
    do_faults = [False, False, True, True]
    superscripts = ['*', '*', '', '']
    subscripts = ['', 't', '']
    runs = [779, 810, 779, 923]
    # runs = [20, 300, 12, 300]
    for i in range(len(do_faults)):
        ax = axs[i % 2]
        stats, controller, Tend = stats_analyser.single_run(
            strategy=BaseStrategy(), run=runs[i], faults=do_faults[i], hook_class=[LogData, LogLocalError], 
        )
        u = get_sorted(stats, type='u')
        faults = get_sorted(stats, type='bitflip')
        for j in range(len(u[0][1])):
            ax.plot(
                [me[0] for me in u],
                [me[1][j] for me in u],
                ls=ls[i],
                color=colors[j],
                label=rf'$u^{{{superscripts[i]}}}_{{{subscripts[j]}}}$',
                marker=markers[0],
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
    plot_fault()
    plot_recovery_rate()
    #plot_vdp_solution()
    plot_efficiency()
    plot_adaptivity_stuff()
    plot_recovery_rate_talk_CSE23()
    # plot_phase_space_things()
