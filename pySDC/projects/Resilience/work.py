import numpy as np
import matplotlib.pyplot as plt


def compare_work(prob, Tend, e_tol=1e-5):
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityRK
    from pySDC.implementations.sweeper_classes.Runge_Kutta import DIRK34
    from pySDC.projects.Resilience.hook import hook_collection, LogData
    from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.convergence_controller_classes.work_model import ReduceWorkInAdaptivity
    from pySDC.helpers.stats_helper import get_sorted

    setup_mpl(reset=True)
    fig, axs = plt.subplots(2, 2, figsize=figsize_by_journal('Springer_Numerical_Algorithms', 1, 0.82), sharex=True)

    problem_params = {}
    problem_params['direct_solver'] = False
    problem_params['newton_tol'] = 1e-8
    problem_params['lintol'] = 1e-8
    # problem_params['mu'] = 10

    shared_desc = {}
    shared_desc['problem_params'] = problem_params
    shared_desc['step_params'] = {'maxiter': 4}

    description = {key: shared_desc[key].copy() for key in shared_desc.keys()}
    description['convergence_controllers'] = {Adaptivity: {'e_tol': e_tol, 'dt_max': 100.0}}

    description_work = {key: description[key].copy() for key in description.keys()}
    description_work['convergence_controllers'][ReduceWorkInAdaptivity] = params

    description_RK = {key: shared_desc[key].copy() for key in shared_desc.keys()}
    description_RK['convergence_controllers'] = {AdaptivityRK: {'e_tol': e_tol, 'dt_max': 1e2, 'update_order': 4}}
    description_RK['sweeper_class'] = DIRK34
    description_RK['step_params'] = {'maxiter': 1}

    description_res = {key: shared_desc[key].copy() for key in shared_desc.keys()}
    description_res['step_params'] = {'maxiter': 99}
    description_res['level_params'] = {'e_tol': e_tol}

    controller_params = {}
    controller_params['hook_class'] = hook_collection + [LogData, LogWork]
    controller_params['logger_level'] = 30

    descriptions = [description, description_work, description_RK, description_res]
    labels = ['adaptivity', 'work-model', 'DIRK4(3)', 'residual']
    S = [
        prob(custom_description=desc, Tend=Tend, custom_controller_params=controller_params)[0] for desc in descriptions
    ]

    ls = {
        0: '-',
        1: '--',
        2: ':',
        3: '-.',
    }
    for i in range(len(S)):
        newton_iter = get_sorted(S[i], type='work_newton')
        axs[0, 0].plot(
            [me[0] for me in newton_iter], np.cumsum([me[1] for me in newton_iter]), label=labels[i], ls=ls[i]
        )

        space_iter = get_sorted(S[i], type='work_linear')
        axs[1, 1].plot([me[0] for me in space_iter], np.cumsum([me[1] for me in space_iter]), ls=ls[i])

        dt = get_sorted(S[i], type='dt', recomputed=False)
        axs[0, 1].plot([me[0] for me in dt], [me[1] for me in dt], ls=ls[i])

        # u = get_sorted(S[i], type='u', recomputed=False)
        # axs[1,1].plot([me[0] for me in u], [max(me[1]) for me in u], ls=ls[i])

        k = get_sorted(S[i], type='sweeps', recomputed=None)
        axs[1, 0].plot([me[0] for me in k], np.cumsum([me[1] for me in k]), ls=ls[i])
    axs[0, 0].set_yscale('log')
    # try:
    #    axs[1, 1].set_yscale('log')
    # except ValueError:
    #    axs[1, 1].set_yscale('linear')
    axs[0, 0].legend(frameon=False)
    axs[0, 0].set_ylabel('cumulative Newton iterations')
    axs[1, 0].set_ylabel('cumulative SDC iterations')
    axs[1, 0].set_xlabel('$t$')
    axs[0, 1].set_ylabel(r'$\Delta t$')
    axs[1, 1].set_ylabel('cumulative GMRES iterations')
    fig.tight_layout()
    fig.savefig('data/work.pdf')


def run_with_limiter(gamma, *args, **kwargs):
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.implementations.convergence_controller_classes.work_model import ReduceWorkInAdaptivity
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.Resilience.leaky_superconductor import run_leaky_superconductor

    Tend = kwargs.get('Tend', 500)

    problem_params = {}
    problem_params['direct_solver'] = False

    description = {}
    description['problem_params'] = problem_params
    description['step_params'] = {'maxiter': 4}

    description['convergence_controllers'] = {
        Adaptivity: {'e_tol': 1e-5, 'dt_max': 100.0},
    }
    if gamma[0] >= 0:
        description['convergence_controllers'][ReduceWorkInAdaptivity] = {
            'gamma': gamma[0],
            'work_keys': kwargs.get('work_keys', ['linear']),
            'stencil_length': 2,
        }

    controller_params = dict()
    controller_params['hook_class'] = [LogWork]

    prob = kwargs.get('prob', run_leaky_superconductor)

    stats, _, T = prob(custom_description=description, Tend=Tend, custom_controller_params=controller_params)

    score = sum([me[1] for me in get_sorted(stats, type=f'work_{kwargs.get("work_keys", ["linear"])[-1]}')])

    print(f'Score={score:.0f}, gamma = {gamma[0]:.4f}')

    if T < Tend:
        score = 1e9
    return score


def optimize():
    from scipy.optimize import minimize
    from pySDC.projects.Resilience.vdp import run_vdp
    import matplotlib.pyplot as plt

    res = {}

    Tend = 10.0
    for ic in np.linspace(0, 1, 49):
        # opt = minimize(run_with_limiter, [ic], tol=1e-2, method='nelder-mead')
        res[ic] = run_with_limiter([ic], prob=run_vdp, Tend=Tend, work_keys=['newton'])
    base_line = run_with_limiter([-1], prob=run_vdp, Tend=Tend, work_keys=['newton'])

    plt.plot(res.keys(), [res[key] for key in res.keys()])
    plt.axhline(base_line)
    plt.show()
    print(res)


if __name__ == '__main__':
    # optimize()
    # raise
    # run_with_limiter([0.2])
    # raise

    from pySDC.projects.Resilience.leaky_superconductor import run_leaky_superconductor

    params = {'work_keys': ['linear', 'newton'], 'gamma': 0.2, 'stencil_length': 2}
    compare_work(run_leaky_superconductor, 500, 1e-5)

    # from pySDC.projects.Resilience.vdp import run_vdp
    # #from pySDC.projects.Resilience.Lorenz import run_Lorenz
    # params = {'work_keys': ['newton'], 'gamma': 1e-3,}
    # compare_work(run_vdp, 10, 1e-6)
