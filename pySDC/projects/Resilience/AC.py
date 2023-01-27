# script to run an Allen-Cahn problem
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.core.Errors import ProblemError
from pySDC.projects.Resilience.hook import LogData, hook_collection
from pySDC.projects.TOMS.AllenCahn_monitor import monitor


class LogErrorAC(monitor):
    def __log_error(self, step, level_number, suffix=''):
        L = step.levels[level_number]
        stats = self.return_stats()
        exact_radius = [me[1] for me in get_sorted(stats, type='exact_radius') if me[0] == L.time + L.dt][0]
        computed_radius = [me[1] for me in get_sorted(stats, type='computed_radius') if me[0] == L.time + L.dt][0]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'e_global{suffix}',
            value=abs(computed_radius - exact_radius),
        )

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self.__log_error(step, level_number, '_post_step')

    def post_run(self, step, level_number):
        super().post_run(step, level_number)
        L = step.levels[level_number]
        e = get_sorted(self.return_stats(), type='e_global_post_step')[-1][1]
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'e_global_post_run',
            value=e,
        )


def run_AC(
    custom_description=None,
    num_procs=1,
    Tend=0.032,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
    use_MPI=False,
    **kwargs,
):
    """
    Run an Allen-Cahn problem with default parameters.

    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        custom_problem_params (dict): Overwrite presets
        use_MPI (bool): Whether or not to use MPI

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-3
    level_params['restol'] = -1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'EE'
    sweeper_params['initial_guess'] = 'spread'

    problem_params = {
        'L': 1.0,
        'init_type': 'circle',
        'nvars': (2**7, 2**7),
        'nu': 2.0,
        'eps': 0.04,
        'radius': 0.25,
    }

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + hook_class if type(hook_class) == list else [hook_class]
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn2d_imex
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if type(custom_description[k]) == dict:
                description[k] = {**description.get(k, {}), **custom_description[k]}
            else:
                description[k] = custom_description[k]

    # set time parameters
    t0 = 0.0

    # instantiate controller
    if use_MPI:
        from mpi4py import MPI
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        comm = kwargs.get('comm', MPI.COMM_WORLD)
        controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)

        # get initial values on finest level
        P = controller.S.levels[0].prob
        uinit = P.u_exact(t0)
    else:
        controller = controller_nonMPI(
            num_procs=num_procs, controller_params=controller_params, description=description
        )

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

    # insert faults
    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        rnd_args = {'iteration': 4, 'problem_pos': (2**6 + 1, 2**6 - 1), 'min_node': 1}
        args = {'time': 0.01, 'target': 0}
        prepare_controller_for_faults(controller, fault_stuff, rnd_args, args)

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except (ProblemError, IndexError):
        stats = controller.return_stats()

    return stats, controller, Tend


def plot_solution(stats, plot_exact=True):  # pragma: no cover
    """
    Plot the radius over time.

    Args:
        stats (dict): The stats object of the run

    Returns:
        None
    """
    r = get_sorted(stats, type='computed_radius')
    r_exact = get_sorted(stats, type='exact_radius')

    # plot sth
    fig, ax = plt.subplots()
    ax.plot([me[0] for me in r], [me[1] for me in r])
    if plot_exact:
        ax.plot([me[0] for me in r_exact], [me[1] for me in r_exact], color='black', ls='--')
    plt.show()


def check_solution(stats, thresh=5e-4):
    """
    Check if the global error solution wrt. a scipy reference solution is tolerable.

    Args:
        stats (dict): The stats object of the run
        thresh (float): Threshold for accepting the accuracy

    Returns:
        None
    """
    error = max([me[1] for me in get_sorted(stats, type='e_global_post_run')])
    assert error < thresh, f"Error too large, got e={error:.2e}"


def main(plotting=True):
    """
    Make a test run and see if the accuracy checks out.

    Args:
        plotting (bool): Plot the solution or not

    Returns:
        None
    """
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun, LogLocalErrorPostStep

    custom_description = {}
    custom_description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-5}}
    custom_controller_params = {'logger_level': 15}
    stats, controller, _ = run_AC(
        custom_description=custom_description,
        custom_controller_params=custom_controller_params,
        hook_class=[LogData, LogErrorAC, LogGlobalErrorPostRun],
    )
    print(sum([me[1] for me in get_sorted(stats, type='sweeps')]))
    check_solution(stats, 5e-4)
    if plotting:  # pragma: no cover
        plot_solution(stats)


if __name__ == "__main__":
    main()
