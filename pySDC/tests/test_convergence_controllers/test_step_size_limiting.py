import pytest


def single_run(useMPI, useWorkLimiter):
    """
    Runs a single advection problem with certain parameters

    Args:
        useMPI (bool): Whether or not to use MPI

    Returns:
       (dict): Stats object generated during the run
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
    from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityExtrapolationWithinQ

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        comm = None

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 0.1
    level_params['restol'] = 1e-10

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['comm'] = comm

    problem_params = {'newton_tol': 1e-12}
    # problem_params = {'freq': 2, 'nvars': 2**9, 'c': 1.0, 'stencil_type': 'center', 'order': 6, 'bc': 'periodic'}

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 99

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogGlobalErrorPostStep, LogWork]
    controller_params['mssdc_jac'] = False

    convergence_controllers = {}
    convergence_controllers[AdaptivityExtrapolationWithinQ] = {'e_tol': 1e-4}
    if useWorkLimiter:
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import WorkLimiter

        convergence_controllers[WorkLimiter] = {}

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=8.0)
    return stats, controller


if __name__ == "__main__":
    from pySDC.helpers.stats_helper import get_sorted, get_list_of_types

    kwargs = {
        'useMPI': False,
    }
    stats_no_limit, controller = single_run(useWorkLimiter=False, **kwargs)
    stats_limit, controller = single_run(useWorkLimiter=True, **kwargs)
    # print(get_list_of_types(stats_limit))
    for stats in [stats_limit, stats_no_limit]:
        niter = sum([me[1] for me in get_sorted(stats, type='niter')])
        newton_iter = sum([me[1] for me in get_sorted(stats, type='work_newton')])

        print(niter, newton_iter)
