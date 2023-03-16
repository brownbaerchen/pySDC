from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pickle

from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.projects.Resilience.Lorenz import run_Lorenz
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.Schroedinger import run_Schroedinger
from pySDC.projects.Resilience.leaky_superconductor import run_leaky_superconductor

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal

setup_mpl(reset=True)
LOGGER_LEVEL = 30
VERBOSE = True

MAPPINGS = {
    'e_global': ('e_global_post_run', max, False),
    'e_global_rel': ('e_global_rel_post_run', max, False),
    't': ('timing_run', max, False),
    'e_local_max': ('e_local_post_step', max, False),
    'k_SDC': ('k', sum, None),
    'k_Newton': ('work_newton', sum, None),
    'k_Newton_no_restart': ('work_newton', sum, False),
    'k_rhs': ('work_rhs', sum, None),
    'restart': ('restart', sum, None),
    'dt_mean': ('dt', np.mean, False),
    'dt_max': ('dt', max, False),
}


def single_run(problem, strategy, data, custom_description, num_procs=1, comm_world=None, problem_args=None):
    """
    Make a single run of a particular problem with a certain strategy.

    Args:
        problem (function): A problem to run
        strategy (Strategy): SDC strategy
        data (dict): Put the results in here
        custom_description (dict): Overwrite presets
        num_procs (int): Number of processes for the time communicator
        comm_world (mpi4py.MPI.Intracomm): Communicator that is available for the entire script

    Returns:
        None
    """
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRunMPI, LogLocalErrorPostStep
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.projects.Resilience.hook import LogData

    strategy_description = strategy.get_custom_description(problem, num_procs)
    description = merge_descriptions(strategy_description, custom_description)

    controller_params = {'logger_level': LOGGER_LEVEL}

    comm = comm_world.Split(comm_world.rank < num_procs)
    problem_args = {} if problem_args is None else problem_args

    if comm_world.rank >= num_procs:
        return None

    stats, controller, _ = problem(
        custom_description=description,
        Tend=strategy.get_Tend(problem, num_procs),
        hook_class=[LogData, LogWork, LogGlobalErrorPostRunMPI, LogLocalErrorPostStep],
        custom_controller_params=controller_params,
        use_MPI=True,
        comm=comm,
        **problem_args,
    )

    # record all the metrics
    for key, mapping in MAPPINGS.items():
        me = get_sorted(stats, type=mapping[0], recomputed=mapping[2])
        if len(me) == 0:
            data[key] += [np.nan]
        else:
            data[key] += [mapping[1]([you[1] for you in me])]
    return None


def get_parameter(dictionary, where):
    """
    Get a parameter at a certain position in a dictionary of dictionaries.

    Args:
        dictionary (dict): The dictionary
        where (list): The list of keys leading to the value you want

    Returns:
        The value of the dictionary
    """
    if len(where) == 1:
        return dictionary[where[0]]
    else:
        return get_parameter(dictionary[where[0]], where[1:])


def set_parameter(dictionary, where, parameter):
    """
    Set a parameter at a certain position in a dictionary of dictionaries

    Args:
        dictionary (dict): The dictionary
        where (list): The list of keys leading to the value you want to set
        parameter: Whatever you want to set the parameter to

    Returns:
        None
    """
    if len(where) == 1:
        dictionary[where[0]] = parameter
    else:
        set_parameter(dictionary[where[0]], where[1:], parameter)


def get_path(problem, strategy, num_procs, handle='', base_path='data/work_precision'):
    """
    Get the path to a certain data.

    Args:
        problem (function): A problem to run
        strategy (Strategy): SDC strategy
        num_procs (int): Number of processes for the time communicator
        handle (str): The name of the configuration
        base_path (str): Some path where all the files are stored

    Returns:
        str: The path to the data you are looking for
    """
    return f'{base_path}/{problem.__name__}-{strategy.__class__.__name__}-{handle}{"-wp" if handle else "wp"}-{num_procs}procs.pickle'


def record_work_precision(
    problem, strategy, num_procs=1, custom_description=None, handle='', runs=1, comm_world=None, problem_args=None
):
    """
    Run problem with strategy and record the cost parameters.

    Args:
        problem (function): A problem to run
        strategy (Strategy): SDC strategy
        num_procs (int): Number of processes for the time communicator
        custom_description (dict): Overwrite presets
        handle (str): The name of the configuration
        runs (int): Number of runs you want to do
        comm_world (mpi4py.MPI.Intracomm): Communicator that is available for the entire script

    Returns:
        None
    """
    data = {}

    # prepare precision parameters
    param = strategy.precision_parameter
    description = merge_descriptions(
        strategy.get_custom_description(problem, num_procs), {} if custom_description is None else custom_description
    )
    if param == 'e_tol':
        power = 10.0
        set_parameter(description, strategy.precision_parameter_loc[:-1] + ['dt_min'], 0)
        set_parameter(description, strategy.precision_parameter_loc[:-1] + ['dt_max'], 5000)
    elif param == 'dt':
        power = 2.0
    elif param == 'restol':
        power = 20 if problem.__name__ == 'run_leaky_superconductor' else 10.0
    else:
        raise NotImplementedError(f"I don't know how to get default value for parameter \"{param}\"")

    where = strategy.precision_parameter_loc
    default = get_parameter(description, where)
    param_range = [default * power**i for i in [-3, -2, -1, 0, 1, 2, 3]]

    if problem.__name__ == 'run_leaky_superconductor':
        if param == 'restol':
            param_range = [1e-4, 1e-5, 1e-6, 1e-7]
        elif param == 'e_tol':
            param_range = [5e-5, 1e-5, 1e-6, 1e-7]

    # run multiple times with different parameters
    for i in range(len(param_range)):
        set_parameter(description, where, param_range[i])
        data[param_range[i]] = {key: [] for key in MAPPINGS.keys()}
        data[param_range[i]]['param'] = [param_range[i]]
        data[param_range[i]][param] = [param_range[i]]
        for _j in range(runs):
            single_run(
                problem,
                strategy,
                data[param_range[i]],
                custom_description=description,
                comm_world=comm_world,
                problem_args=problem_args,
            )

            comm_world.Barrier()

            if VERBOSE and comm_world.rank == 0:
                print(
                    f'{problem.__name__} {handle}, {param}={param_range[i]:.2e}: e={data[param_range[i]]["e_global"][-1]}, t={data[param_range[i]]["t"][-1]}, k={data[param_range[i]]["k_SDC"][-1]}'
                )

    if comm_world.rank == 0:
        with open(get_path(problem, strategy, num_procs, handle), 'wb') as f:
            pickle.dump(data, f)


def plot_work_precision(
    problem,
    strategy,
    num_procs,
    ax,
    work_key='k_SDC',
    precision_key='e_global',
    handle='',
    plotting_params=None,
    comm_world=None,
):
    """
    Plot data from running a problem with a strategy.

    Args:
        problem (function): A problem to run
        strategy (Strategy): SDC strategy
        num_procs (int): Number of processes for the time communicator
        ax (matplotlib.pyplot.axes): Somewhere to plot
        work_key (str): The key in the recorded data you want on the x-axis
        precision_key (str): The key in the recorded data you want on the y-axis
        handle (str): The name of the configuration
        plotting_params (dict): Will be passed when plotting
        comm_world (mpi4py.MPI.Intracomm): Communicator that is available for the entire script

    Returns:
        None
    """
    if comm_world.rank > 0:
        return None

    with open(get_path(problem, strategy, num_procs, handle=handle), 'rb') as f:
        data = pickle.load(f)

    work = [np.nanmean(data[key][work_key]) for key in data.keys()]
    precision = [np.nanmean(data[key][precision_key]) for key in data.keys()]

    for key in [work_key, precision_key]:
        rel_variance = [np.std(data[me][key]) / np.nanmean(data[me][key]) for me in data.keys()]
        if not all([me < 1e-2 or not np.isfinite(me) for me in rel_variance]):
            print(f"WARNING: Variance too large! Got {rel_variance}")

    style = merge_descriptions(
        {**strategy.style, 'label': f'{strategy.style["label"]}{f" {handle}" if handle else ""}'},
        plotting_params if plotting_params else {},
    )

    ax.loglog(work, precision, **style)


def decorate_panel(ax, problem, work_key, precision_key, num_procs=1, title_only=False):
    """
    Decorate a plot

    Args:
        ax (matplotlib.pyplot.axes): Somewhere to plot
        problem (function): A problem to run
        work_key (str): The key in the recorded data you want on the x-axis
        precision_key (str): The key in the recorded data you want on the y-axis
        num_procs (int): Number of processes for the time communicator
        title_only (bool): Put only the title on top, or do the whole shebang

    Returns:
        None
    """
    labels = {
        'k_SDC': 'SDC iterations',
        'k_Newton': 'Newton iterations',
        'k_Newton_no_restart': 'Newton iterations (restarts excluded)',
        'k_rhs': 'right hand side evaluations',
        't': 'wall clock time / s',
        'e_global': 'global error',
        'e_global_rel': 'relative global error',
        'e_local_max': 'max. local error',
        'restart': 'restarts',
        'dt_max': r'$\Delta t_\mathrm{max}$',
        'param': 'parameter',
    }

    if not title_only:
        ax.set_xlabel(labels.get(work_key, 'work'))
        ax.set_ylabel(labels.get(precision_key, 'precision'))
        # ax.legend(frameon=False)

    titles = {
        'run_vdp': 'Van der Pol',
        'run_Lorenz': 'Lorenz attractor',
        'run_Schroedinger': r'Schr\"odinger',
        'run_leaky_superconductor': 'Quench',
    }
    ax.set_title(titles.get(problem.__name__, ''))


def execute_configurations(
    problem, configurations, work_key, precision_key, num_procs, ax, decorate, record, runs, comm_world
):
    """
    Run for multiple configurations.

    Args:
        problem (function): A problem to run
        configurations (dict): The configurations you want to run with
        work_key (str): The key in the recorded data you want on the x-axis
        precision_key (str): The key in the recorded data you want on the y-axis
        num_procs (int): Number of processes for the time communicator
        ax (matplotlib.pyplot.axes): Somewhere to plot
        decorate (bool): Whether to decorate fully or only put the title
        record (bool): Whether to only plot or also record the data first
        runs (int): Number of runs you want to do
        comm_world (mpi4py.MPI.Intracomm): Communicator that is available for the entire script

    Returns:
        None
    """
    for _, config in configurations.items():
        for strategy in config['strategies']:
            shared_args = {
                'problem': problem,
                'strategy': strategy,
                'handle': config.get('handle', ''),
                'num_procs': config.get('num_procs', num_procs),
            }
            if record:
                record_work_precision(
                    **shared_args,
                    custom_description=config.get('custom_description', {}),
                    runs=runs,
                    comm_world=comm_world,
                    problem_args=config.get('problem_args', {}),
                )
            plot_work_precision(
                **shared_args,
                work_key=work_key,
                precision_key=precision_key,
                ax=ax,
                plotting_params=config.get('plotting_params', {}),
                comm_world=comm_world,
            )

    decorate_panel(
        ax=ax,
        problem=problem,
        work_key=work_key,
        precision_key=precision_key,
        num_procs=num_procs,
        title_only=not decorate,
    )


def get_configs(mode, problem):
    # TODO: docs
    configurations = {}
    if mode == 'compare_strategies':
        from pySDC.projects.Resilience.strategies import AdaptivityStrategy, BaseStrategy, IterateStrategy

        description_high_order = {'step_params': {'maxiter': 5}}
        description_low_order = {'step_params': {'maxiter': 3}}
        dashed = {'ls': '--'}

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

        description_large_step = {
            'level_params': {'dt': 30.0 if problem.__name__ == 'run_leaky_superconductor' else 3e-2}
        }
        description_small_step = {
            'level_params': {'dt': 10.0 if problem.__name__ == 'run_leaky_superconductor' else 1e-2}
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
    elif mode == 'compare_adaptivity':
        # TODO: configurations not final!
        from pySDC.projects.Resilience.strategies import (
            AdaptivityCollocationTypeStrategy,
            AdaptivityCollocationRefinementStrategy,
            AdaptivityStrategy,
        )

        strategies = [AdaptivityCollocationTypeStrategy(), AdaptivityCollocationRefinementStrategy()]

        restol = None
        for strategy in strategies:
            strategy.restol = restol

        configurations[1] = {'strategies': strategies, 'handle': '*'}
        configurations[2] = {
            'custom_description': {'step_params': {'maxiter': 5}},
            'strategies': [AdaptivityStrategy()],
        }

        # strategies2 = [AdaptivityCollocationTypeStrategy(), AdaptivityCollocationRefinementStrategy()]
        # restol = 1e-6
        # for strategy in strategies2:
        #    strategy.restol = restol
        # configurations[3] = {'strategies':strategies2, 'handle': 'low restol', 'plotting_params': {'ls': '--'}}

    elif mode == 'quench':
        from pySDC.projects.Resilience.strategies import (
            AdaptivityStrategy,
            DoubleAdaptivityStrategy,
            IterateStrategy,
            BaseStrategy,
        )

        dumbledoresarmy = DoubleAdaptivityStrategy()
        # dumbledoresarmy.residual_e_tol_ratio = 1e2
        dumbledoresarmy.residual_e_tol_abs = 1e-3

        strategies = [AdaptivityStrategy(), IterateStrategy(), BaseStrategy(), dumbledoresarmy]
        configurations[1] = {'strategies': strategies}
        configurations[2] = {
            'strategies': strategies,
            'problem_args': {'imex': True},
            'handle': 'IMEX',
            'plotting_params': {'ls': '--'},
        }
        inexact = {'problem_params': {'newton_iter': 30}}
        configurations[3] = {
            'strategies': strategies,
            'custom_description': inexact,
            'handle': 'inexact',
            'plotting_params': {'ls': ':'},
        }
        LU = {'sweeper_params': {'QI': 'LU'}}
        configurations[4] = {
            'strategies': strategies,
            'custom_description': LU,
            'handle': 'LU',
            'plotting_params': {'ls': '-.'},
        }
    elif mode == 'preconditioners':
        from pySDC.projects.Resilience.strategies import AdaptivityStrategy, IterateStrategy, BaseStrategy

        strategies = [AdaptivityStrategy(), IterateStrategy(), BaseStrategy()]

        precons = ['IE', 'LU', 'MIN', 'MIN3']
        ls = ['-', '--', '-.', ':']
        for i in range(len(precons)):
            configurations[i] = {
                'strategies': strategies,
                'custom_description': {'sweeper_params': {'QI': precons[i]}},
                'handle': precons[i],
                'plotting_params': {'ls': ls[i]},
            }
    else:
        raise NotImplementedError(f'Don\'t know the mode "{mode}"!')

    return configurations


def get_fig(x=1, y=1, **kwargs):
    # TODO: docs
    keyword_arguments = {
        'figsize': figsize_by_journal('Springer_Numerical_Algorithms', 1.0, 1.0),
        'layout': 'constrained',
        **kwargs,
    }
    return plt.subplots(x, y, **keyword_arguments)


def save_fig(fig, name, work_key, precision_key, legend=True, format='pdf', **kwargs):
    """
    Save a figure with a legend on the bottom.

    Args:
        fig (matplotlib.pyplot.Figure): Figure you want to save
        name (str): Name of the plot to put in the path
        work_key (str): The key in the recorded data you want on the x-axis
        precision_key (str): The key in the recorded data you want on the y-axis
        legend (bool): Put a legend or not
        format (str): Format to store the figure with

    Returns:
        None
    """
    handles, labels = fig.get_axes()[0].get_legend_handles_labels()
    order = np.argsort([me[0] for me in labels])
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order], loc='outside lower center', ncols=2, frameon=False
    )

    path = f'data/wp-{name}-{work_key}-{precision_key}.{format}'
    fig.savefig(path, bbox_inches='tight', **kwargs)
    print(f'Stored figure \"{path}\"')


def all_problems(mode='compare_strategies', **kwargs):
    """
    Make a plot comparing various strategies for all problems.

    Args:
        work_key (str): The key in the recorded data you want on the x-axis
        precision_key (str): The key in the recorded data you want on the y-axis

    Returns:
        None
    """

    fig, axs = get_fig(2, 2)

    shared_params = {
        'work_key': 'k_SDC',
        'precision_key': 'e_global',
        'num_procs': 1,
        'runs': 1,
        'comm_world': MPI.COMM_WORLD,
        'record': False,
        **kwargs,
    }

    problems = [run_vdp, run_Lorenz, run_Schroedinger, run_leaky_superconductor]

    for i in range(len(problems)):
        execute_configurations(
            **shared_params,
            problem=problems[i],
            ax=axs.flatten()[i],
            decorate=i == 2,
            configurations=get_configs(mode, problems[i]),
        )

    save_fig(
        fig=fig,
        name=mode,
        work_key=shared_params['work_key'],
        precision_key=shared_params['precision_key'],
        legend=True,
    )


def single_problem(mode, problem, **kwargs):
    """
    Make a plot for a single problem

    Args:
        mode (str): What you want to look at
        problem (function): A problem to run
    """
    fig, ax = get_fig(1, 1)

    params = {
        'work_key': 'k_SDC',
        'precision_key': 'e_global',
        'num_procs': 1,
        'runs': 1,
        'comm_world': MPI.COMM_WORLD,
        'record': False,
        **kwargs,
    }

    execute_configurations(**params, problem=problem, ax=ax, decorate=True, configurations=get_configs(mode, problem))

    save_fig(
        fig=fig,
        name=f'{problem.__name__}-{mode}',
        work_key=params['work_key'],
        precision_key=params['precision_key'],
        legend=False,
    )


if __name__ == "__main__":
    comm_world = MPI.COMM_WORLD
    # params = {
    #    'mode': 'preconditioners',
    #    'problem': run_vdp,
    #    'runs': 5,
    # }
    # record = True
    # single_problem(**params, work_key='t', precision_key='e_global_rel', record=record, runs=1)

    all_params = {
        'record': True,
        'runs': 5,
        'work_key': 't',
        'precision_key': 'e_global_rel',
    }

    for mode in ['compare_strategies', 'preconditioners', 'compare_adaptivity']:
        all_problems(**all_params, mode=mode)
        comm_world.Barrier()
    plt.show()
