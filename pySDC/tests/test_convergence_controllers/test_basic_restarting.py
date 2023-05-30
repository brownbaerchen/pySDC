import pytest


def run_problem(
    maxiter=1,
    num_procs=1,
    n_steps=1,
    error_estimator=None,
    params=None,
    restol=-1,
    restarts=None,
    dt=0.1,
    min_dt=None,
    max_restarts=10,
    restart_from_first_step=False,
    crash_after_max_restarts=True,
    useMPI=False,
    spread_from_first_restarted=True,
    **kwargs,
):
    import numpy as np
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.hooks.log_errors import (
        LogLocalErrorPostIter,
        LogGlobalErrorPostIter,
        LogLocalErrorPostStep,
    )
    from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestarting
    from pySDC.implementations.convergence_controller_classes.spread_step_sizes import SpreadStepSizesBlockwise
    from pySDC.core.Hooks import hooks
    from pySDC.core.ConvergenceController import ConvergenceController

    class artificial_restarts(ConvergenceController):
        def __init__(self, controller, params, description, **kwargs):
            super().__init__(controller, params, description, **kwargs)
            self.restart_times = [0.6] if restarts is None else restarts.copy()
            self._encountered_restart = False

        def determine_restart(self, controller, S, **kwargs):
            super().determine_restart(controller, S, **kwargs)

            if S.status.iter < S.params.maxiter and S.levels[0].status.residual > S.levels[0].params.restol:
                return None

            if any(abs(me - S.time) < dt / 10.0 for me in self.restart_times):
                S.status.restart = True
                self.restart_times.pop(np.argmin([abs(me - S.time) for me in self.restart_times]))
                self._encountered_restart = True
                self.logger.info(f'Triggering a restart on process {S.status.slot} at t={S.time:.1e}')
            else:
                self._encountered_restart = False

            if useMPI:
                updated = kwargs['comm'].allgather(self._encountered_restart)
                if any(updated):
                    first_updated = np.min(np.arange(len(updated))[updated])
                    self.restart_times = kwargs['comm'].bcast(self.restart_times, root=first_updated)

    class artificial_adaptivity(hooks):
        def __init__(self):
            super().__init__()
            self.min_dt = [] if min_dt is None else min_dt.copy()

        def post_iteration(self, step, level_number):
            super().post_iteration(step, level_number)

            if (step.status.iter == maxiter or step.level.status.residual <= step.level.restol) and len(
                self.min_dt
            ) > 0:

                step.levels[0].status.dt_new = dt * (1 + abs(step.status.slot - self.min_dt[0]))

                if step.status.last:
                    self.min_dt.remove(self.min_dt[0])
                self.logger.info(f'{step.status.slot} get\'s dt_new = {step.levels[0].status.dt_new:.2e}')

    # initialize level parameters
    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = restol

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    # sweeper_params['initial_guess'] = 'random'

    # build lambdas
    re = np.linspace(-30, -1, 10)
    im = np.linspace(-50, 50, 11)
    lambdas = np.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
        (len(re) * len(im))
    )

    problem_params = {
        'lambdas': lambdas,
        'u0': 1.0 + 0.0j,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # convergence controllers
    BasicRestartingParams = {
        "max_restarts": max_restarts,
        "crash_after_max_restarts": bool(crash_after_max_restarts),
        "restart_from_first_step": bool(restart_from_first_step),
    }
    SpreadStepSizesParams = {
        "spread_from_first_restarted": bool(spread_from_first_restarted),
    }
    convergence_controllers = {
        BasicRestarting.get_implementation(useMPI=useMPI): BasicRestartingParams,
        SpreadStepSizesBlockwise.get_implementation(useMPI=useMPI): SpreadStepSizesParams,
        artificial_restarts: {},
    }

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 11
    controller_params['hook_class'] = [artificial_adaptivity]
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = testequation0d
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    # set time parameters
    t0 = 0.0

    # instantiate controller
    if useMPI:
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        controller = controller_MPI(comm=comm, controller_params=controller_params, description=description)
        P = controller.S.levels[0].prob
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

        controller = controller_nonMPI(
            num_procs=num_procs, controller_params=controller_params, description=description
        )
        P = controller.MS[0].levels[0].prob

    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=n_steps * level_params['dt'])
    return stats


# @pytest.mark.base
# def test_spread_step_size(num_procs=4, restart_from_first_step=False, spread_from_first_restarted=True):
#     from pySDC.helpers.stats_helper import get_sorted
#     import numpy as np
#
#     dt = 1.e-1
#     n_steps = 24
#     min_dt = [2, 1, 3]
#     planned_restarts = [0.6, 1.0]
#
#     stats = run_problem(num_procs=num_procs, n_steps=n_steps, restarts=planned_restarts, min_dt=min_dt, dt=dt, restart_from_first_step=restart_from_first_step, spread_from_first_restarted=spread_from_first_restarted)


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_procs', [1, 4, 3])
@pytest.mark.parametrize('restart_from_first_step', [0, 1])
def test_basic_restarting_MPI(num_procs, restart_from_first_step):
    import os
    import subprocess

    kwargs = {}
    kwargs['useMPI'] = 1
    kwargs['num_procs'] = num_procs
    kwargs['restart_from_first_step'] = restart_from_first_step

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    # run code with different number of MPI processes
    kwargs_str = "".join([f"{key}:{item} " for key, item in kwargs.items()])
    cmd = f"mpirun -np {num_procs} python {__file__} {kwargs_str}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )


@pytest.mark.base
@pytest.mark.parametrize('num_procs', [1, 3, 4])
@pytest.mark.parametrize('restart_from_first_step', [False, True])
def test_basic_restarting_nonMPI(num_procs, restart_from_first_step):
    basic_restarting_single_test(useMPI=False, num_procs=num_procs, restart_from_first_step=restart_from_first_step)


def basic_restarting_single_test(**kwargs):
    from pySDC.helpers.stats_helper import get_sorted
    import numpy as np

    arguments = {
        'useMPI': False,
        'restarts': [0.6, 1.0],
        'dt': 1.0e-1,
        'n_steps': 24,
        'num_procs': 1,
        "max_restarts": 10,
        "crash_after_max_restarts": True,
        "restart_from_first_step": False,
        **kwargs,
    }

    stats = run_problem(**arguments)

    if arguments['useMPI']:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    # compute the time blocks
    num_blocks = arguments['n_steps'] // arguments['num_procs']
    block_starts = np.array([me * arguments['dt'] * arguments['num_procs'] for me in range(num_blocks)])

    restarts = get_sorted(stats, type='restart', comm=comm)
    times_restarted = [me[0] for me in restarts if me[1]]

    # compute expected restarts
    expected_restarts = []
    for me in arguments['restarts']:
        if me > arguments['n_steps'] * arguments['dt']:
            continue

        # figure out the block I happened in
        my_block_start = max(block_starts[block_starts <= me + arguments['dt'] / 10.0])
        my_block = [my_block_start + arguments['dt'] * i for i in range(arguments['num_procs'])]

        if arguments['restart_from_first_step']:
            expected_restarts += my_block
        else:
            expected_restarts += [t for t in my_block if t >= me]

    times_restarted = sorted(times_restarted)
    expected_restarts = sorted(expected_restarts)

    # make sure we got everything we expected
    assert all(
        np.isclose(times_restarted[i], expected_restarts[i], atol=arguments['dt'] / 10)
        for i in range(len(expected_restarts))
    ), f"Didn\'t get the restarts we expected! Got {times_restarted}, expected {expected_restarts}"


if __name__ == "__main__":
    import sys

    kwargs = {me.split(':')[0]: int(me.split(':')[1]) for me in sys.argv[1:]}
    basic_restarting_single_test(**kwargs)
