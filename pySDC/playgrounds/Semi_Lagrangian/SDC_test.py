from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.playgrounds.Semi_Lagrangian.BurgersSL import (
    Burgers1DSL,
    AdvectionDiffusion1DSL,
    AdvectionDiffusion1D,
    AdvectionDiffusion1DIMEX,
    Burgers1DIMEX,
)
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.playgrounds.Semi_Lagrangian.implicit_semi_lagrangian import implicit_SL, generic_implicit
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types


def run(dt=1e-1, nsteps=1, problem_class=AdvectionDiffusion1DSL):
    sweeper_classes = {
        AdvectionDiffusion1D: generic_implicit,
        AdvectionDiffusion1DIMEX: imex_1st_order,
        Burgers1DIMEX: imex_1st_order,
    }

    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1e-12

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 4
    sweeper_params['QI'] = 'MIN-SR-S'

    problem_params = {'nu': 1e-2}

    step_params = {}
    step_params['maxiter'] = 99

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogSolution, LogGlobalErrorPostStep]

    description = {}
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_classes.get(problem_class, implicit_SL)
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(controller_params=controller_params, description=description, num_procs=1)

    P = controller.MS[0].levels[0].prob
    t0 = 0.0
    u0 = P.u_exact(0)
    Tend = nsteps * dt

    uend, stats = controller.run(u0=u0, t0=t0, Tend=Tend)
    return stats, P


def run_problem(dt=1e-1, nsteps=10, problem_class=AdvectionDiffusion1DIMEX):
    stats, P = run(dt=dt, nsteps=nsteps, problem_class=problem_class)
    u = get_sorted(stats, type='u')
    fig = P.get_fig()
    for me in u:
        P.plot(me[1], me[0], fig)

    u_exact = P.u_exact(u[-1][0])
    P.plot(u_exact, t=u[-1][0], fig=fig, ls='--')


def compare_advection():
    dts = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    errors = []
    errors_FI = []
    errors_IMEX = []
    for dt in dts:
        stats, _ = run(dt)
        errors.append(get_sorted(stats, type='e_global_post_step')[-1][1])

        stats_FI, _ = run(dt, problem_class=AdvectionDiffusion1D)
        errors_FI.append(get_sorted(stats_FI, type='e_global_post_step')[-1][1])

        stats_IMEX, _ = run(dt, problem_class=AdvectionDiffusion1DIMEX)
        errors_IMEX.append(get_sorted(stats_IMEX, type='e_global_post_step')[-1][1])
    plt.loglog(dts, errors, label='Semi Lagrangian')
    plt.loglog(dts, errors_FI, label='Fully implicit')
    plt.loglog(dts, errors_IMEX, label='IMEX')
    for power in [3, 4]:
        plt.loglog(dts, [me**power for me in dts], label=f'{power}th order', ls='--')
    plt.legend(frameon=False)
    plt.xlabel('step size')
    plt.ylabel('local error')


def compare_Burgers():
    dts = [5e-1, 1e-1, 5e-2]
    Tend = 0.5
    errors = {'SL': [], 'IMEX': []}
    stats_ref, _ = run(dt=min(dts) / 10, nsteps=Tend / (min(dts) / 10), problem_class=Burgers1DIMEX)
    u_ref = get_sorted(stats_ref, type='u')[-1][1]

    for dt in dts:
        stats, _ = run(dt, Tend / dt, problem_class=Burgers1DSL)
        u = get_sorted(stats, type='u')[-1][1]
        errors['SL'].append(abs(u - u_ref))

        stats, _ = run(dt, Tend / dt, problem_class=Burgers1DIMEX)
        u = get_sorted(stats, type='u')[-1][1]
        errors['IMEX'].append(abs(u - u_ref))

    plt.loglog(dts, errors['SL'], label='Semi Lagrangian')
    plt.loglog(dts, errors['IMEX'], label='IMEX')
    for power in [1, 2, 3, 4]:
        plt.loglog(dts, [me**power for me in dts], label=f'{power}th order', ls='--')
    plt.legend(frameon=False)
    plt.xlabel('step size')
    plt.ylabel('global error')


if __name__ == '__main__':
    from qmat.lagrange import LagrangeApproximation

    nodes = [0, 1, 2, 3]
    L = LagrangeApproximation(points=nodes)
    Q = L.getIntegrationMatrix(intervals=[[0, 1]])
    breakpoint()
    # compare_Burgers()
    # compare_advection()
    # run_problem(dt=0.5, nsteps=1, problem_class=Burgers1DSL)
    plt.show()
