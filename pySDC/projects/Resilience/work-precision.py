import numpy as np
import matplotlib.pyplot as plt
import pickle

from pySDC.projects.Resilience.strategies import AdaptivityStrategy, IterateStrategy, BaseStrategy, merge_descriptions
from pySDC.projects.Resilience.Lorenz import run_Lorenz
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.Schroedinger import run_Schroedinger
from pySDC.projects.Resilience.leaky_superconductor import run_leaky_superconductor

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal

setup_mpl(reset=True)
LOGGER_LEVEL = 30
VERBOSE = True


def single_run(problem, strategy, data, custom_description, num_procs=1):
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun, LogLocalErrorPostStep
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.projects.Resilience.hook import LogData

    strategy_description = strategy.get_custom_description(problem, num_procs)
    description = merge_descriptions(strategy_description, custom_description)

    controller_params = {'logger_level': LOGGER_LEVEL}

    stats, controller, _ = problem(
        custom_description=description,
        Tend=strategy.get_Tend(problem, num_procs),
        hook_class=[LogData, LogWork, LogGlobalErrorPostRun, LogLocalErrorPostStep],
        custom_controller_params=controller_params,
    )

    # record all the metrics
    mappings = {
        'e_global': ('e_global_post_run', max, False),
        't': ('timing_run', max, False),
        'e_local_max': ('e_local_post_step', max, False),
        'k_SDC': ('k', sum, None),
        'k_Newton': ('work_newton', sum, None),
        'k_rhs': ('work_rhs', sum, None),
    }
    for key, mapping in mappings.items():
        me = get_sorted(stats, type=mapping[0], recomputed=mapping[2])
        if len(me) == 0:
            data[key] += [np.nan]
        else:
            data[key] += [mapping[1]([you[1] for you in me])]
    return None


def get_parameter(dictionary, where):
    if len(where) == 1:
        return dictionary[where[0]]
    else:
        return get_parameter(dictionary[where[0]], where[1:])


def set_parameter(dictionary, where, parameter):
    if len(where) == 1:
        dictionary[where[0]] = parameter
    else:
        set_parameter(dictionary[where[0]], where[1:], parameter)


def get_path(problem, strategy, handle='', base_path='data/work_precision'):
    return f'{base_path}/{problem.__name__}-{strategy.__class__.__name__}-{handle}{"-wp" if handle else "wp"}.pickle'


def record_work_precision(problem, strategy, num_procs=1, custom_description=None, handle='', runs=1):
    data = {}

    # prepare precision parameters
    param = strategy.precision_parameter
    description = merge_descriptions(
        strategy.get_custom_description(problem, num_procs), {} if custom_description is None else custom_description
    )
    if param == 'e_tol':
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        where = ['convergence_controllers', Adaptivity, 'e_tol']
        power = 1.5 if problem.__name__ == 'run_leaky_superconductor' else 10.0
        set_parameter(description, ['convergence_controllers', Adaptivity, 'dt_min'], 0)
    elif param == 'dt':
        where = ['level_params', 'dt']
        power = 2.0
    elif param == 'restol':
        where = ['level_params', 'restol']
        power = 20 if problem.__name__ == 'run_leaky_superconductor' else 10.0
    else:
        raise NotImplementedError(f"I don't know how to get default value for parameter \"{param}\"")

    default = get_parameter(description, where)
    param_range = [default * power**i for i in [-2, -1, 0, 1, 2]]

    if problem.__name__ == 'run_leaky_superconductor' and param == 'restol':
        param_range = [1e-5, 1e-6, 1e-7, 1e-8]

    # run multiple times with different parameters
    for i in range(len(param_range)):
        set_parameter(description, where, param_range[i])
        data[param_range[i]] = {key: [] for key in ['e_global', 'e_local_max', 'k_SDC', 'k_Newton', 'k_rhs', 't']}
        for _j in range(runs):
            single_run(problem, strategy, data[param_range[i]], custom_description=description)

            if VERBOSE:
                print(
                    f'{problem.__name__} {handle}, {param}={param_range[i]:.2e}: e={data[param_range[i]]["e_global"][-1]}, t={data[param_range[i]]["t"][-1]}, k={data[param_range[i]]["k_SDC"][-1]}'
                )

    with open(get_path(problem, strategy, handle), 'wb') as f:
        pickle.dump(data, f)


def plot_work_precision(
    problem,
    strategy,
    num_procs,
    ax,
    work_key='k_SDC',
    precision_key='e_global',
    reload=True,
    handle='',
    plotting_params=None,
):
    with open(get_path(problem, strategy, handle=handle), 'rb') as f:
        data = pickle.load(f)

    work = [np.nanmean(data[key][work_key]) for key in data.keys()]
    precision = [np.nanmean(data[key][precision_key]) for key in data.keys()]

    style = merge_descriptions(
        {**strategy.style, 'label': f'{strategy.style["label"]}{f" {handle}" if handle else ""}'},
        plotting_params if plotting_params else {},
    )

    ax.loglog(work, precision, **style)


def decorate_panel(ax, problem, work_key, precision_key, num_procs=1, title_only=False):
    xlabels = {
        'k_SDC': 'SDC iterations',
        'k_Newton': 'Newton iterations',
        'k_rhs': 'right hand side evaluations',
        't': 'wall clock time / s',
    }

    ylabels = {
        'e_global': 'global error',
        'e_local_max': 'max. local error',
    }

    if not title_only:
        ax.set_xlabel(xlabels.get(work_key, 'work'))
        ax.set_ylabel(ylabels.get(precision_key, 'precision'))
        ax.legend(frameon=False)

    titles = {
        'run_vdp': 'Van der Pol',
        'run_Lorenz': 'Lorenz attractor',
        'run_Schroedinger': r'Schr\"odinger',
        'run_leaky_superconductor': 'Quench',
    }
    ax.set_title(titles.get(problem.__name__, ''))


def execute_configurations(problem, work_key, precision_key, num_procs, ax, decorate, record, runs):
    description_high_order = {'step_params': {'maxiter': 5}}
    description_low_order = {'step_params': {'maxiter': 3}}
    description_large_step = {'level_params': {'dt': 10.0 if problem.__name__ == 'run_leaky_superconductor' else 3e-2}}
    description_small_step = {'level_params': {'dt': 5.0 if problem.__name__ == 'run_leaky_superconductor' else 1e-2}}

    dashed = {'ls': '--'}

    configurations = {}
    configurations[0] = {
        'custom_description': description_high_order,
        'handle': r'high order',
        'strategies': [AdaptivityStrategy(), BaseStrategy()],
    }
    configurations[1] = {
        'custom_description': description_low_order,
        'handle': r'low order',
        'strategies': [AdaptivityStrategy(), BaseStrategy()],
        'plotting_params': dashed,
    }
    configurations[2] = {
        'custom_description': description_large_step,
        'handle': r'large step',
        'strategies': [IterateStrategy()],
        'plotting_params': dashed,
    }
    configurations[3] = {
        'custom_description': description_small_step,
        'handle': r'small step',
        'strategies': [IterateStrategy()],
    }

    for _, config in configurations.items():
        for strategy in config['strategies']:
            shared_args = {
                'problem': problem,
                'strategy': strategy,
                'handle': config.get('handle', ''),
                'num_procs': config.get('num_procs', num_procs),
            }
            if record:
                record_work_precision(**shared_args, custom_description=config.get('custom_description', {}), runs=runs)
            plot_work_precision(
                **shared_args,
                work_key=work_key,
                precision_key=precision_key,
                ax=ax,
                plotting_params=config.get('plotting_params', {}),
            )

    decorate_panel(
        ax=ax,
        problem=problem,
        work_key=work_key,
        precision_key=precision_key,
        num_procs=num_procs,
        title_only=not decorate,
    )


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2, figsize=figsize_by_journal('Springer_Numerical_Algorithms', 1.0, 1.0))

    shared_params = {
        'work_key': 'k_SDC',
        'precision_key': 'e_global',
        'num_procs': 1,
        'runs': 10,
        'record': True,
    }

    problems = [run_vdp, run_Lorenz, run_Schroedinger, run_leaky_superconductor]

    for i in range(len(problems)):
        execute_configurations(**shared_params, problem=problems[i], ax=axs.flatten()[i], decorate=i == 2)

    fig.tight_layout()
    plt.show()
