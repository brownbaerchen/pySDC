import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

from pySDC.playgrounds.Preconditioners.hooks import log_cost
from pySDC.playgrounds.Preconditioners.configs import (
    get_params,
    store_precon,
    store_serial_precon,
    get_collocation_nodes,
    prepare_sweeper,
)

print_status = False


def single_run(x, params, *args, **kwargs):
    '''
    This function takes as input the diagonal preconditioner entries and runs a problem.
    The args should contain the problem to run in position 0

    Args:
        x (numpy.ndarray): The entries of the preconditioner
        params (dict): Parameters for setting up the run
        convergence_controllers (dict): Convergence controllers to use

    Returns:
        dict: Stats of the run
        pySDC.controller: The controller used in the run
    '''

    # setup adaptivity and problem parameters
    custom_description = {
        'convergence_controllers': params.get('convergence_controllers', {}),
    }

    problem_params = params['problem_params']

    controller_params = params.get('controller_params', {})
    controller_params['logger_level'] = controller_params.get('logger_level', 30)

    sweeper_params, sweeper = prepare_sweeper(x, params, **kwargs)
    custom_description['sweeper_params'] = {
        **params.get('sweeper_params', {}),
        **sweeper_params,
        **params.get('force_sweeper_params', {}),
        **kwargs.get('force_sweeper_params', {}),
    }
    custom_description['sweeper_class'] = kwargs.get('force_sweeper', sweeper)

    allowed_keys = ['step_params', 'level_params']
    for key in allowed_keys:
        custom_description[key] = {**custom_description.get(key, {}), **params.get(key, {})}

    stats, controller, _ = params['prob'](
        custom_description=custom_description,
        hook_class=kwargs.get('hook_class', log_cost),
        custom_problem_params=problem_params,
        custom_controller_params=controller_params,
        Tend=params['Tend'],
        **kwargs.get('pkwargs', {}),
    )
    return stats, controller


def get_defaults(x, params):
    """
    Run the problem with LU to see how many iterations that requires

    Args:
        x (numpy.ndarray): The entries of the preconditioner (ignored except for inferring the number of nodes)
        params (dict): Params that are passed to `single_run`

    Returns:
        None
    """
    stats, controller = single_run(
        x, params, force_sweeper=generic_implicit, force_sweeper_params={'QI': 'LU', 'num_nodes': 3}
    )
    params['k'] = sum([me[1] for me in get_sorted(stats, type='k', recomputed=None)])

    if print_status:
        print(f'Needed {params["k"]} iterations for {params["name"]} problem with LU')


def objective_function_k_only(x, *args):
    '''
    This function takes as input the diagonal preconditioner entries and runs a problem and then returns the number of
    iterations.

    The args should contain the params for a problem in position 0

    Args:
        x (numpy.ndarray): The entries of the preconditioner

    Returns:
        int: Number of iterations
    '''
    params = args[0]
    kwargs = args[1]

    stats, controller = single_run(x, params, *args, **kwargs)

    # check how many iterations we needed
    k = sum([me[1] for me in get_sorted(stats, type='k')])

    # get error
    e = get_error(stats, controller)

    # return the score
    score = k
    if print_status:
        print(f's={score:7.0f} | e={e:.2e} | k: {k - params["k"]:5} | sum(x)={sum(x):.2f}', x)

    return score


def get_error(stats, controller):
    """
    Get the error at the end of a pySDC run

    Args:
        stats (pySDC.stats): Stats object generated by a pySDC run
        controller (pySDC.controller): Controller used for the run

    Returns:
        float: Error at the end of the run
    """
    u_end = get_sorted(stats, type='u')[-1]
    exact = controller.MS[0].levels[0].prob.u_exact(t=u_end[0])
    return abs(u_end[1] - exact)


def optimize(params, initial_guess, num_nodes, objective_function, tol=1e-16, **kwargs):
    """
    Run a single optimization run and store the result

    Args:
        params (dict): Parameters for running the problem
        initial_guess (numpy.ndarray): Initial guess to start the minimization problem
        num_nodes (int): Number of collocation nodes
        objective_function (function): Objective function for the minimizaiton alogrithm

    Returns:
        None
    """
    get_defaults(initial_guess, params)
    if kwargs.get('use_complex'):
        ics = np.zeros(len(initial_guess) * 2)
        ics[::2] = initial_guess
        initial_guess = ics
    if kwargs.get('SOR'):
        store_precon(params, initial_guess, initial_guess, **kwargs)
    else:
        opt = minimize(objective_function, initial_guess, args=(params, kwargs), tol=tol, method='nelder-mead')
        store_precon(params, opt.x, initial_guess, **kwargs)


def objective_function_k_and_e(x, *args):
    '''
    This function takes as input the diagonal preconditioner entries and runs a problem and then returns the number of
    iterations.

    The args should contain the problem parameters in position 0

    Args:
        x (numpy.ndarray): The entries of the preconditioner

    Returns:
        int: Number of iterations
    '''
    raise NotImplementedError
    params = args[0]

    stats, controller = single_run(x, params)

    # check how many iterations we needed
    k = sum([me[1] for me in get_sorted(stats, type='k')])

    # check if we solved the problem correctly
    u_end = get_sorted(stats, type='u')[-1]
    exact = controller.MS[0].levels[0].prob.u_exact(t=u_end[0])
    e = abs(exact - u_end[1])
    e_em = max([me[1] for me in get_sorted(stats, type='e_em', recomputed=False)])
    raise NotImplementedError('Please fix the next couple of lines')

    # return the score
    score = k * e / args[6]
    print(f's={score:7.0f} | k: {k - args[5]:5} | e: {e / args[6]:.2e} | e_em: {e_em / args[7]:.2e}', x)
    # print(x, k, f'e={e:.2e}', f'e_em={e_em:.2e}')

    return score


def plot_errors(stats, u_end, exact):
    plt.plot(np.arange(len(u_end[1])), u_end[1])
    u_old = get_sorted(stats, type='u_old')[-1]
    plt.plot(np.arange(len(exact)), exact, ls='--')
    error = np.abs(exact - u_end[1])
    plt.plot(np.arange(len(exact)), error, label=f'e_max={error.max():.2e}')
    plt.plot(np.arange(len(u_old[1])), np.abs(u_old[1] - u_end[1]), label='e_em')
    # plt.yscale('log')
    plt.legend(frameon=False)
    plt.pause(1e-9)
    plt.cla()


def optimize_with_sum(params, num_nodes, **kwargs):
    initial_guess = (np.arange(num_nodes - 1) + 1) / (num_nodes + 2)
    kwargs['normalized'] = True
    optimize(params, initial_guess, num_nodes, objective_function_k_only, **kwargs)


def optimize_diagonal(params, num_nodes, **kwargs):
    initial_guess = np.array(get_collocation_nodes(params, num_nodes)) * 0.7
    optimize(params, initial_guess, num_nodes, objective_function_k_only, **kwargs)


def optimize_with_first_row(params, num_nodes, **kwargs):
    i0 = np.array(get_collocation_nodes(params, num_nodes)) / 2 * 1.3
    initial_guess = np.append(i0, i0)
    # initial_guess = np.append(np.ones(num_nodes) * 0.9, - (0.9 - i0 * 2 + np.finfo(float).eps))
    kwargs['use_first_row'] = True
    optimize(params, initial_guess, num_nodes, objective_function_k_only, **kwargs)


if __name__ == '__main__':
    print_status = True
    problem = 'Dahlquist'

    kwargs = {
        'adaptivity': True,
        'random_IG': True,
        # 'use_complex': True,
        #'SOR': True,
    }

    params = get_params(problem, **kwargs)
    num_nodes = 3

    store_serial_precon(problem, num_nodes, LU=True, **kwargs)
    store_serial_precon(problem, num_nodes, IE=True, **kwargs)
    store_serial_precon(problem, num_nodes, IEpar=True, **kwargs)
    store_serial_precon(problem, num_nodes, MIN=True, **kwargs)
    store_serial_precon(problem, num_nodes, MIN3=True, **kwargs)

    optimize_diagonal(params, num_nodes, **kwargs)
    optimize_with_first_row(params, num_nodes, **kwargs)
    # optimize_with_sum(params, num_nodes, **kwargs)
