# script to run a quench problem
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit, allencahn_semiimplicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.projects.Resilience.hook import hook_collection, LogData
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.projects.Resilience.sweepers import imex_1st_order_efficient, generic_implicit_efficient
import numpy as np

from pySDC.core.Errors import ConvergenceError


def run_AC(
    custom_description=None,
    num_procs=1,
    Tend=1e-2,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    imex=False,
    u0=None,
    t0=None,
    use_MPI=False,
    **kwargs,
):
    """
    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        imex (bool): Solve the problem IMEX or fully implicit
        u0 (dtype_u): Initial value
        t0 (float): Starting time
        use_MPI (bool): Whether or not to use MPI

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """
    if custom_description is not None:
        problem_params = custom_description.get('problem_params', {})
        if 'imex' in problem_params.keys():
            imex = problem_params['imex']
            problem_params.pop('imex', None)

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 1e-4
    level_params['restol'] = 1e-8

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'newton_tol': 1e-9,
    }

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = allencahn_semiimplicit if imex else allencahn_fullyimplicit
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order_efficient if imex else generic_implicit_efficient
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        description = merge_descriptions(description, custom_description)

    # set time parameters
    t0 = 0.0 if t0 is None else t0

    # instantiate controller
    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }
    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = kwargs.get('comm', MPI.COMM_WORLD)
        controller = controller_MPI(**controller_args, comm=comm)
        P = controller.S.levels[0].prob
    else:
        controller = controller_nonMPI(**controller_args, num_procs=num_procs)
        P = controller.MS[0].levels[0].prob

    uinit = P.u_exact(t0) if u0 is None else u0

    # insert faults
    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        prepare_controller_for_faults(controller, fault_stuff)

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ConvergenceError:
        print('Warning: Premature termination!')
        stats = controller.return_stats()
    return stats, controller, Tend


def plot_solution(stats):  # pragma: no cover
    import matplotlib.pyplot as plt
    from pySDC.helpers.stats_helper import get_sorted

    fig, ax = plt.subplots(1, 1)

    u = get_sorted(stats, type='u', recomputed=False)
    print(get_sorted(stats, type='e_local_post_step', recomputed=False))
    for me in u:  # pun intended
        ax.imshow(me[1], vmin=-1, vmax=1)
        ax.set_title(f't={me[0]:.2e}')
        plt.pause(1e-1)

    plt.show()


if __name__ == '__main__':
    from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep

    stats, _, _ = run_AC(imex=True, hook_class=LogLocalErrorPostStep)
    plot_solution(stats)
