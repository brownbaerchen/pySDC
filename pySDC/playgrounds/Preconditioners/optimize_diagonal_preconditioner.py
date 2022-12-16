import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

from pySDC.playgrounds.Preconditioners.hooks import log_cost
from pySDC.playgrounds.Preconditioners.configs import (
    get_params,
    store_precon,
    store_serial_precon,
    get_collocation_nodes,
    prepare_sweeper,
    get_optimization_initial_conditions,
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
    custom_description['sweeper_class'] = sweeper
    custom_description['max_restarts'] = 20

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


def get_defaults(x, params, QI='LU'):
    """
    Run the problem with LU to see how many iterations that requires

    Args:
        x (numpy.ndarray): The entries of the preconditioner (ignored except for inferring the number of nodes)
        params (dict): Params that are passed to `single_run`

    Returns:
        None
    """
    stats, controller = single_run(
        x,
        params,
        force_sweeper=generic_implicit,
        force_sweeper_params={'QI': QI, 'num_nodes': params['sweeper_params'].get('num_nodes', 3)},
        **kwargs,
    )
    params['k'] = sum([me[1] for me in get_sorted(stats, type='k', recomputed=None)])
    params['residual_post_step'] = max([me[1] for me in get_sorted(stats, type='residual_post_step', recomputed=False)])
    params['residual_post_step'] = get_sorted(stats, type='residual_post_step', recomputed=False)[-1][1]

    if print_status:
        print(f'Needed {params["k"]} iterations for {params["name"]} problem with {QI}')


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
    k = sum([me[1] for me in get_sorted(stats, type='k', recomputed=None)])

    # get error
    e = get_error(stats, controller)

    # return the score
    score = k
    if print_status:
        print(f's={score:7.0f} | e={e:.2e} | k: {k - params["k"]:5} | sum(x)={sum(x):.2f}', x)

    return score


def objective_function_k_and_residual(x, *args):
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
    k = sum([me[1] for me in get_sorted(stats, type='k', recomputed=None)])

    # check the residual
    r = get_sorted(stats, type='residual_post_step', recomputed=None)[-1][1]

    # return the score
    score = k * max([r / params['residual_post_step'], 1.])
    if print_status:
        print(f'opt iter: {params["opt_iter"]} | s={score:7.0f} | r={r:.2e} | k: {k - params["k"]:5}', x)

    params["opt_iter"] += 1

    return score


def objective_function_k_and_restarts(x, *args):
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
    k = sum([me[1] for me in get_sorted(stats, type='k', recomputed=None)])
    restarts = get_sorted(stats, type='restarts', recomputed=None)
    times = np.unique([me[0] for me in restarts])
    restart = [sum([me[1] for me in restarts if me[0]==t]) for t in times]
    #print(restart)

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


def optimizeMC(params, num_nodes, objective_function, tol=1e-8, repeat_optimization=1, num_samples=200, skip_fac=10, **kwargs):
    """
    Run a series of optimizations to obtain results 'independent' of initial conditions of the optimization
    """
    # get sample initial conditions to infer the structure of the initial conditions
    opt_IC_shape = get_optimization_initial_conditions(params, num_nodes, **kwargs).shape

    # initalize a candidate dictionary
    candidates = {
        'optimization_IC': dict(),
        'optimization_score': dict(),
        'optimization_result': dict(),
    }

    # initalize a random generator
    rng = np.random.RandomState(kwargs.get('random_seed', 1984))

    # initalize a highscore to beat
    highscore = np.inf

    from pySDC.playgrounds.Preconditioners.configs import get_name
    for n in range(num_samples):

        # generate initial conditions for the optimization
        IC = rng.rand(*opt_IC_shape)
        IC_orig = IC.copy()
        print(f' --- MC IC: # {n}: ', IC, '---')

        # determine if we want to try these initial conditions at all
        if skip_fac:
            if params['k'] == 0:
                get_defaults(IC, params, QI='LU')
            score = objective_function_k_only(IC, params, kwargs)
            if score > skip_fac * params['k']:
                print(f'Score {score} exceeds {skip_fac} times the iterations needed with LU ({params["k"]}), skipping these initial conditions')
                candidates['optimization_IC'][n] = IC_orig
                candidates['optimization_score'][n] = score
                candidates['optimization_result'][n] = IC
                continue

        # perform the optimization, possibly multiple times
        for i in range(repeat_optimization):
            kwargs['initial_conditions'] = IC
            IC_new = optimize(params, num_nodes, objective_function, tol, no_store=True, **kwargs)
            if np.allclose(IC_new, IC, rtol=1e-2):
                IC = IC_new
                print(f'Minimum reached after {i} optimizations, moving on to next initial guess')
                break
            IC = IC_new

        score = objective_function(IC, params, kwargs)
        candidates['optimization_IC'][n] = IC_orig
        candidates['optimization_score'][n] = score
        candidates['optimization_result'][n] = IC

        # store the preconditioner as the total optimization result if the highscore was beat
        if score < highscore:
            highscore = score
            store_precon(params, IC, 'MC', **{**kwargs, 'initial_conditions': 'MC'})

    # store the optimization result
    name = get_name(params['name'], num_nodes, **kwargs)
    with open(f'data/precons/optimization-candidates-{name}.pickle', 'wb') as file:
        pickle.dump(candidates, file)


def optimize(params, num_nodes, objective_function, tol=1e-12, **kwargs):
    """
    Run a single optimization run and store the result

    Args:
        params (dict): Parameters for running the problem
        initial_conditions (numpy.ndarray) or (str): Initial guess to start the minimization problem
        num_nodes (int): Number of collocation nodes
        objective_function (function): Objective function for the minimizaiton alogrithm

    Returns:
        numpy.ndarray: Optimization results
    """
    if not 'initial_conditions' in kwargs.keys():
        raise ValueError('Need "initial_conditions" in keyword arguments to start optimization')
    if type(kwargs['initial_conditions']) == str:
        if kwargs.get('SOR', False):
            ics = {
                'MIN3': 1.0,
                'IEpar': 0.7,
                'MIN': 1.25,
            }
            opt_IC = np.array([ics.get(kwargs['initial_conditions'], 1.0)])
        else:
            opt_IC = get_optimization_initial_conditions(params, num_nodes, **kwargs)
        get_defaults(opt_IC, params, QI=kwargs['initial_conditions'])
    else:
        opt_IC = kwargs['initial_conditions']
        get_defaults(opt_IC, params, QI='LU')


    if kwargs.get('use_complex'):
        ics = np.zeros(len(opt_IC) * 2)
        ics[::2] = opt_IC
        opt_IC = ics

    params['opt_iter'] = 0
    opt = minimize(objective_function, opt_IC, args=(params, kwargs), tol=tol, method='nelder-mead')

    if not kwargs.get('no_store', False):
        store_precon(params, opt.x, opt_IC, **kwargs)

    return opt.x


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
    e_em = max([me[1] for me in get_sorted(stats, type='e_em', recomputed=None)])
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


def run_optimization(problem, num_nodes, args, kwargs):

    params = get_params(problem, **kwargs)
    params['sweeper_params']['num_nodes'] = num_nodes

    for k in args.keys():
        for v in args[k]:
            kwargs[k] = v
            for k2 in args.keys():
                if k == k2:
                    break
                for v2 in args[k2]:
                    kwargs[k2] = v2
                    optimize(params=params, num_nodes=num_nodes, objective_function=objective_function_k_and_restarts, **kwargs)


def run_optimizationMC(problem, num_nodes, kwargs, num_samples=200, repeat_optimization=1):
    params = get_params(problem, **kwargs)
    params['sweeper_params']['num_nodes'] = num_nodes
    optimizeMC(params=params, num_nodes=num_nodes, objective_function=objective_function_k_and_residual, num_samples=num_samples, repeat_optimization=repeat_optimization, **kwargs)


if __name__ == '__main__':
    print_status = True

    problem = 'Dahlquist'
    num_nodes = 3

    args = {
        'initial_conditions': ['MIN', 'MIN3', 'IEpar'],
        'use_first_row': [False, True],
        #'SOR': [True, False],
    }

    kwargs = {
        'adaptivity': True,
        'initial_guess': 'spread',
        #'initial_conditions': 'IEpar',
        # 'use_complex': True,
        #'SOR': True,
    }

    for precon in ['LU', 'IE', 'IEpar', 'MIN', 'MIN3']:
        store_serial_precon(problem, num_nodes, **{precon: True})

    run_optimizationMC(problem, num_nodes, kwargs, num_samples=1000, repeat_optimization=10)
    #run_optimization(problem, num_nodes, args, kwargs)
