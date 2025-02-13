# script to run a ParaDiag with advection equation
from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

import numpy as np


def get_description(dt):
    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1e-6

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['initial_guess'] = 'copy'

    problem_params = {'nvars': 64, 'order': 8, 'c': 1}

    step_params = {}
    step_params['maxiter'] = 9

    convergence_controllers = {}

    description = {}
    description['problem_class'] = advectionNd
    description['problem_params'] = problem_params
    description['sweeper_class'] = QDiagonalization
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    return description


def get_controller_params():

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogGlobalErrorPostRun]
    controller_params['alpha'] = 1e-4

    return controller_params


def run_advection(
    n_steps=4,
    dt=0.1,
):
    description = get_description(dt)
    controller_params = get_controller_params()

    controller = controller_ParaDiag_nonMPI(
        num_procs=n_steps, description=description, controller_params=controller_params
    )

    for S in controller.MS:
        S.levels[0].prob.init = tuple([*S.levels[0].prob.init[:2]] + [np.dtype('complex128')])

    P = controller.MS[0].levels[0].prob

    t0 = 0.0
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=n_steps * dt)


if __name__ == '__main__':
    n_steps = 12
    run_advection(n_steps=n_steps, dt=1e-1)
