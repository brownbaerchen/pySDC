from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pickle

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError

from pySDC.projects.Resilience.hook import log_timings
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.heat import run_heat


problem_names = {
    run_advection: 'Advection',
    run_vdp: 'Van der Pol',
    run_heat: 'Diffusion',
}


def record_timing(problem, sizes, **kwargs):
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

        stats = run(problem, comm, **kwargs)

        # record the timing for the entire run
        for k in record_keys:
            res[k][comm.size] = ops.get(k, np.mean)([me[1] for me in get_sorted(stats, type=keys.get(k, k), comm=comm)])

    if MPI.COMM_WORLD.rank == 0:
        name = get_name(problem, **kwargs)
        with open(f"data/{name}.pickle", 'wb') as file:
            pickle.dump(res, file)
        plot_timing(problem, **kwargs)
    return res


def run(problem, comm=None, adaptivity=False, Tend=5.0, smooth=None):
    # TODO: docs
    custom_controller_params = {'logger_level': 30}
    custom_description = {'convergence_controllers': {}}

    if adaptivity:
        custom_description['convergence_controllers'][Adaptivity] = {'e_tol': 1e-7}
    else:
        custom_description['convergence_controllers'][EstimateEmbeddedError.get_implementation('MPI')] = {}

    if smooth == True:
        custom_description['problem_params'] = {
            'freq': 2,
        }
    elif smooth == False:
        custom_description['problem_params'] = {
            'freq': -1,
            'sigma': 0.6,
        }

    # run the problem
    stats, controller, Tend = problem(
        custom_description=custom_description,
        num_procs=comm.size,
        use_MPI=True,
        custom_controller_params=custom_controller_params,
        Tend=Tend,
        comm=comm,
        hook_class=log_timings,
    )
    return stats


def plot(problem, **kwargs):
    stats = run(problem, MPI.COMM_WORLD, Tend=0.5, **kwargs)
    e_embedded = [me[1] for me in get_sorted(stats, type='e_embedded', recomputed=False, comm=MPI.COMM_WORLD)]
    t = [me[0] for me in get_sorted(stats, type='e_embedded', recomputed=False, comm=MPI.COMM_WORLD)]
    if MPI.COMM_WORLD.rank == 0:
        plt.plot(t, e_embedded)
        # plt.yscale('log')
        plt.show()
    # print(MPI.COMM_WORLD.rank, np.min(e_embedded), np.max(e_embedded), np.std(e_embedded), np.mean(e_embedded))


def get_name(problem, **kwargs):
    """
    Get a unique identifier for a certain configuration for storing and loading.

    Args:
        problem (function): A function that runs a pySDC problem

    Returns:
        str: The identifier
    """
    name = f'{problem_names[problem]}-timings'
    for k in kwargs.keys():
        name = f'{name}{f"-with-{k}" if kwargs[k] else f"-no-{k}"}'
    return name


def plot_timing(problem, cluster='.', **kwargs):
    # TODO: docs
    if MPI.COMM_WORLD.rank != 0:
        return None

    name = get_name(problem, **kwargs)
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

    ax.set_title(name.replace('-', ' '))

    fig.tight_layout()

    plt.savefig(f"data/{name}.pdf")
    plt.show()


if __name__ == "__main__":
    problem = run_advection
    sizes = np.arange(MPI.COMM_WORLD.size) + 1
    cluster = 'juwels'

    kwargs = {
        'adaptivity': True,
        'smooth': False,
    }

    record_timing(problem, sizes, **kwargs)
    # plot_timing(problem, cluster, **kwargs)
    # plot(problem, **kwargs)
