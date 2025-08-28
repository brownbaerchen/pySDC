from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.playgrounds.Semi_Lagrangian.BurgersSL import Burgers1DSL, AdvectionDiffusion1DSL
from pySDC.playgrounds.Semi_Lagrangian.implicit_semi_lagrangian import implicit_SL
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types


def run(dt=1e-1, nsteps=1):
    level_params = {}
    level_params['dt'] = dt

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 4
    sweeper_params['QI'] = 'MIN-SR-S'

    problem_params = {}

    step_params = {}
    step_params['maxiter'] = 4

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogSolution, LogGlobalErrorPostStep]

    description = {}
    description['problem_class'] = Burgers1DSL
    description['problem_params'] = problem_params
    description['sweeper_class'] = implicit_SL
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


stats, P = run(nsteps=10)
u = get_sorted(stats, type='u')
fig = P.get_fig()
for me in u:
    P.plot(me[1], me[0], fig)


u_exact = P.u_exact(u[-1][0])
P.plot(u_exact, t=u[-1][0], fig=fig, ls='--')

# dts = [4e-1, 3e-1, 2e-1, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 1e-2, 5e-3, 1e-3]
# errors = []
# for dt in dts:
#     stats, _ = run(dt)
#
#     errors.append(get_sorted(stats, type='e_global_post_step')[-1][1])
# print(errors)
# plt.loglog(dts, errors)
# for power in [1, 2, 3, 4]:
#     plt.loglog(dts, [me**power for me in dts], label=power, ls='--')
# plt.legend(frameon=False)


plt.show()
