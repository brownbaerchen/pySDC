from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pickle

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

from pySDC.projects.Resilience.hook import log_timings
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.heat import run_heat


problem_names = {
    run_advection: 'Advection',
    run_vdp: 'Van der Pol',
    run_heat: 'Diffusion',
}


def record_timing(problem, sizes, adaptivity):
    # TODO: docs
    res = {}
    res['problem'] = problem

    record_keys = [
        'timing_run',
        'timing_setup',
        'timing_iteration',
        'timing_comm',
        'timing_step',
        'dt',
        'niter',
        'total_iterations',
        'nsteps',
    ]

    ops = {
        'total_iterations': sum,
        'nsteps': len,
    }
    keys = {
        'total_iterations': 'niter',
        'nsteps': 'niter',
    }

    for k in record_keys:
        res[k] = {}

    for s in sizes:
        comm = MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank < s)

        # all ranks that don't participate just immediately skip to the next iteration
        if MPI.COMM_WORLD.rank >= s:
            continue

        custom_description = {}
        custom_controller_params = {'logger_level': 30}

        if adaptivity:
            custom_description = {'convergence_controllers': {}}
            custom_description['convergence_controllers'][Adaptivity] = {'e_tol': 1e-7}

        # run the problem
        stats, controller, Tend = problem(
            custom_description=custom_description,
            num_procs=comm.size,
            use_MPI=True,
            custom_controller_params=custom_controller_params,
            Tend=5.0,
            comm=comm,
            hook_class=log_timings,
        )

        # record the timing for the entire run
        for k in record_keys:
            res[k][comm.size] = ops.get(k, np.mean)([me[1] for me in get_sorted(stats, type=keys.get(k, k), comm=comm)])

    if MPI.COMM_WORLD.rank == 0:
        name = get_name(problem, adaptivity)
        with open(f"data/{name}.pickle", 'wb') as file:
            pickle.dump(res, file)
        plot_timing(problem, adaptivity)
    return res


def get_name(problem, adaptivity):
    # TODO: docs
    return f'{problem_names[problem]}-timings{"-with-adaptivity" if adaptivity else ""}'


def plot_timing(problem, adaptivity, cluster='.'):
    # TODO: docs
    if MPI.COMM_WORLD.rank != 0:
        return None

    name = get_name(problem, adaptivity)
    with open(f"data/{cluster}/{name}.pickle", 'rb') as file:
        res = pickle.load(file)

    fig, ax = plt.subplots()

    plotting_keys = ['timing_run']
    sizes = np.unique(list(res['timing_run'].keys()))

    for k in plotting_keys:
        ax.plot(sizes, [res[k][s] for s in sizes], label=k)

    k_dict = res.get('niter', {'niter': 5})
    k = np.mean([k_dict[me] for me in k_dict.keys()])
    n = np.array(sizes)
    ax.loglog(sizes, res['timing_run'][1] / n, color='black', linestyle='--', label='ideal speedup')
    # ax.loglog(sizes, res['timing_run'][1] * (n - 1 + k) / n / k, color='black', linestyle='-.', label='maximal speedup')
    # ax.loglog(sizes, [res['timing_run'][1] * res['total_iterations'][s] / res['total_iterations'][1] / s for s in sizes] , color='grey', linestyle='-.', label='maximal speedup')

    ax.legend(frameon=False)
    ax.set_xlabel('MPI size')
    ax.set_ylabel('Wall clock time')

    timing = [res['timing_run'][s] for s in sizes]
    print(min(timing), max(timing))
    print(np.argmin(timing), np.argmax(timing))

    ax.set_title(f'{problem_names[problem]}{" with adaptivity" if adaptivity else ""}')

    fig.tight_layout()

    plt.savefig(f"data/{name}.pdf")
    plt.show()


if __name__ == "__main__":
    problem = run_advection
    sizes = np.arange(MPI.COMM_WORLD.size) + 1
    adaptivity = True
    cluster = 'juwels'

    # record_timing(problem, sizes, adaptivity)
    plot_timing(problem, adaptivity, cluster)
