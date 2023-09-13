import pytest


def get_controller(dt, num_nodes, quad_type, useMPI, high_Taylor_order, **kwargs):
    """
    Runs a single advection problem with certain parameters

    Args:
        dt (float): Step size
        num_nodes (int): Number of nodes
        quad_type (str): Type of quadrature
        useMPI (bool): Whether or not to use MPI
        high_Taylor_order (bool): Whether to use the low order approximation or the high order one

    Returns:
       (dict): Stats object generated during the run
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
        EstimateExtrapolationErrorWithinQ,
    )

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        comm = None

    # initialize level parameters
    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1.0

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = quad_type
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['comm'] = comm

    problem_params = {'degree': 20}

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 0

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = polynomial_testequation
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {
        EstimateExtrapolationErrorWithinQ: {'high_Taylor_order': high_Taylor_order}
    }

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    return controller


def single_test(**kwargs):
    """
    Run a single test where the solution is replaced by a polynomial and the nodes are changed.
    Because we know the polynomial going in, we can check if the interpolation based change was
    exact. If the solution is not a polynomial or a polynomial of higher degree then the number
    of nodes, the change in nodes does add some error, of course, but here it is on the order of
    machine precision.
    """
    import numpy as np

    args = {
        'num_nodes': 3,
        'quad_type': 'RADAU-RIGHT',
        'useMPI': False,
        'high_Taylor_order': True,
        'dt': 1.0,
        **kwargs,
    }

    # prepare variables
    controller = get_controller(**args)
    step = controller.MS[0]
    level = step.levels[0]
    prob = level.prob
    cont = controller.convergence_controllers[
        np.arange(len(controller.convergence_controllers))[
            [type(me).__name__ == 'EstimateExtrapolationErrorWithinQ' for me in controller.convergence_controllers]
        ][0]
    ]
    nodes = np.append([0], level.sweep.coll.nodes)

    # initialize variables
    step.status.slot = 0
    step.status.iter = 1
    level.status.time = 0.0
    level.status.residual = 0.0
    level.u[0] = prob.u_exact(t=0)
    level.sweep.predict()

    for i in range(len(level.u)):
        if level.u[i] is not None:
            level.u[i][:] = prob.u_exact(nodes[i] * level.dt)

    # perform the interpolation
    cont.post_iteration_processing(controller, step)
    error = level.status.error_extrapolation_estimate

    return error


def single_run(dt, Tend, num_nodes, quad_type, QI, useMPI, high_Taylor_order):
    """
    Runs a single advection problem with certain parameters

    Args:
        dt (float): Step size
        Tend (float): Final time
        num_nodes (int): Number of nodes
        quad_type (str): Type of quadrature
        QI (str): Preconditioner
        useMPI (bool): Whether or not to use MPI
        high_Taylor_order (bool): Whether to use the low order approximation or the high order one

    Returns:
       (dict): Stats object generated during the run
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
    from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
        EstimateExtrapolationErrorWithinQ,
    )

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        comm = None

    # initialize level parameters
    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1e-10

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = quad_type
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['QI'] = QI
    sweeper_params['comm'] = comm

    problem_params = {'freq': 2, 'nvars': 2**9, 'c': 1.0, 'stencil_type': 'center', 'order': 6, 'bc': 'periodic'}

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 99

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = LogGlobalErrorPostStep
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = advectionNd
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {
        EstimateExtrapolationErrorWithinQ: {'high_Taylor_order': high_Taylor_order}
    }

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller


def multiple_runs(dts, **kwargs):
    """
    Make multiple runs of a specific problem and record vital error information

    Args:
        dts (list): The step sizes to run with
        num_nodes (int): Number of nodes
        quad_type (str): Type of nodes

    Returns:
        dict: Errors for multiple runs
        int: Order of the collocation problem
    """
    from pySDC.helpers.stats_helper import get_sorted

    res = {}

    for dt in dts:
        # stats, controller = single_run(Tend=5.0 * dt, dt=dt, **kwargs)

        res[dt] = {}
        # res[dt]['e_loc'] = max([me[1] for me in get_sorted(stats, type='e_global_post_step')])
        # res[dt]['e_ex'] = max([me[1] for me in get_sorted(stats, type='error_extrapolation_estimate')])
        res[dt]['e'] = single_test(dt=dt, **kwargs)

    print(res)
    coll_order = -1  # controller.MS[0].levels[0].sweep.coll.order
    return res, coll_order


def check_order(dts, **kwargs):
    """
    Check the order by calling `multiple_runs` and then `plot_and_compute_order`.

    Args:
        dts (list): The step sizes to run with
        num_nodes (int): Number of nodes
        quad_type (str): Type of nodes
    """
    import numpy as np

    res, coll_order = multiple_runs(dts, **kwargs)
    dts = np.array(list(res.keys()))
    keys = list(res[dts[0]].keys())

    expected_order = {
        'e': kwargs['num_nodes'] + 1 if kwargs['high_Taylor_order'] else kwargs['num_nodes'],
        'e_loc': coll_order + 1,
        'e_ex': kwargs['num_nodes'] + 1 if kwargs['high_Taylor_order'] else kwargs['num_nodes'],
    }

    for key in keys:
        errors = np.array([res[dt][key] for dt in dts])

        mask = np.logical_and(errors < 1e-0, errors > 1e-14)
        order = np.log(errors[mask][1:] / errors[mask][:-1]) / np.log(dts[mask][1:] / dts[mask][:-1])

        assert np.isclose(
            np.mean(order), expected_order[key], atol=0.5
        ), f'Expected order {expected_order[key]} for {key}, but got {np.mean(order):.2e}!'


@pytest.mark.base
@pytest.mark.parametrize('num_nodes', [2, 3, 4, 5])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
@pytest.mark.parametrize('high_Taylor_order', [False])
def test_extrapolation_within_Q(num_nodes, quad_type, high_Taylor_order):
    kwargs = {
        'num_nodes': num_nodes,
        'quad_type': quad_type,
        'useMPI': False,
        'QI': 'MIN',
        'high_Taylor_order': high_Taylor_order,
    }
    steps = [1 / 2.0**i for i in [3, 4, 5, 6]]
    steps = [1 / 2.0**i for i in [1, 2, 3, 4, 5, 6]]
    import numpy as np

    steps = np.logspace(1, -4, 20)
    check_order(steps, **kwargs)


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_nodes', [2, 5])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
def test_extrapolation_within_Q_MPI(num_nodes, quad_type):
    import subprocess
    import os

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_nodes} python {__file__} {num_nodes} {quad_type}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_nodes,
    )


if __name__ == "__main__":
    import sys

    print(single_test(dt=0.5 / 2.0))

    if len(sys.argv) > 1:
        kwargs = {
            'num_nodes': int(sys.argv[1]),
            'quad_type': sys.argv[2],
            'useMPI': True,
            'QI': 'MIN',
            'high_Taylor_order': True,
        }
        check_order([5e-1, 1e-1, 8e-2, 5e-2], **kwargs)