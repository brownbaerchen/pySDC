from mpi4py import MPI
import logging
import numpy as np
from pySDC.projects.Resilience.work_precision import record_work_precision, get_configs, run_vdp
from pySDC.projects.Resilience.strategies import AdaptivityExtrapolationWithinQStrategy, merge_descriptions
from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityExtrapolationWithinQ

logger = logging.getLogger('Optimization')
logger.setLevel(26)


def punisher_of_worlds(data):
    return [np.array(data[key]['k_Newton']) * np.array(data[key]['e_global']) for key in data.keys() if key != 'meta']


def constructor_of_mars(x):
    newton_maxiter = int(abs(x[0] * 1e3))
    maxiter = int(abs(x[1] * 1e3))
    e_tol_rel = abs(x[2])

    description = {}
    description['problem_params'] = {'newton_maxiter': newton_maxiter}
    description['step_params'] = {'maxiter': maxiter}

    adaptivity_params = {
        'e_tol_rel': e_tol_rel,
        'e_tol': 1e-7,
    }
    description['convergence_controllers'] = {AdaptivityExtrapolationWithinQ: adaptivity_params}
    return description


config = {
    'problem': run_vdp,
    'strategy': AdaptivityExtrapolationWithinQStrategy(useMPI=True),
    'work_key': 'work_newton',
    'precision_key': 'e_global',
    'num_procs': 1,
    'comm_world': MPI.COMM_WORLD,
    'Tend': None,
    'param_range': [1e-7],
    'problem_args': {},
    'custom_description': {},
    'num_procs_sweeper': 1,
    'hooks': None,
    'runs': 1,
    'mode': '',
    'objective_function': punisher_of_worlds,
    'parameter_function': constructor_of_mars,
}


def single_run(
    x=None,
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
        plotting (bool): Whether to plot something
        num_procs_sweeper (int): Number of processes for the sweeper
        mode (str): What you want to look at

    Returns:
        None
    """
    custom_description = config.get('custom_description', {})
    parameter_function = config['parameter_function']
    objective_function = config['objective_function']

    kwargs_names = [
        'problem',
        'strategy',
        'num_procs',
        'num_procs_sweeper',
        'comm_world',
        'mode',
        'param_range',
        'runs',
    ]
    params = {key: value for key, value in config.items() if key in kwargs_names}

    custom_description = merge_descriptions(custom_description, parameter_function(x))
    if config['comm_world'].rank == 0:
        logger.log(25, f'Starting with parameters {x}')
    data = record_work_precision(
        **params,
        custom_description=custom_description,
    )
    score = np.mean(objective_function(data))
    if config['comm_world'].rank == 0:
        logger.log(26, f'Got {score} for parameters {x}')
    return score


# single_run(x = [1])
from scipy.optimize import minimize, basinhopping, dual_annealing

fun = single_run
x0 = [1e-1, 1e-1, 1e-1]  # [ 0.52733565, -0.63271077, -1.85010228]
bounds = [(1e-3, 1e-1), (1e-3, 1e-1), (1e-7, 1e-2)]
# x0 = [1e-1]
# bounds = [(1e-7, 1e-2)]
maxiter = 6

# opt = dual_annealing(fun, x0=x0, bounds=bounds, maxiter=maxiter)
# opt = minimize(fun, x0=x0, tol=1e-3, bounds=bounds, maxiter=maxiter)
opt = basinhopping(fun, x0=x0)
breakpoint()
