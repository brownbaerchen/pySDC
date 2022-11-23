from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pickle

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

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

    record_keys = ['timing_run', 'timing_setup', 'timing_iteration', 'timing_comm', 'timing_step']

    for k in record_keys:
        res[k] = {}

    for s in sizes:
        comm = MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank < s)

        # all ranks that don't participate just immediately skip to the next iteration
        if MPI.COMM_WORLD.rank >= s:
            continue

        custom_description = {}
        custom_controller_params = {}

        if adaptivity:
            custom_description = {'convergence_controllers': {}}
            custom_description['convergence_controllers'][Adaptivity] = {'e_tol': 1e-7}

        # run the problem
        stats, controller, Tend = problem(
            custom_description=custom_description,
            num_procs=comm.size,
            use_MPI=True,
            custom_controller_params=custom_controller_params,
            Tend=10.0,
            comm=comm,
        )

        # record the timing fot the entire run
        for k in record_keys:
            res[k][comm.size] = np.mean([me[1] for me in get_sorted(stats, type=k)])

    if MPI.COMM_WORLD.rank == 0:
        # plot some stuff
        fig, ax = plt.subplots()

        plotting_keys = ['timing_run']

        for k in plotting_keys:
            ax.plot(sizes, [res[k][s] for s in sizes], label=k)

        ax.loglog(sizes, res['timing_run'][1] / np.array(sizes), color='black', linestyle='--', label='ideal speedup')

        ax.legend(frameon=False)
        ax.set_xlabel('MPI size')
        ax.set_ylabel('Wall clock time')

        ax.set_title(f'{problem_names[problem]}{" with adaptivity" if adaptivity else ""}')

        fig.tight_layout()

        name = f'{problem_names[problem]}-timings-{"-with-adaptivity" if adaptivity else ""}'
        plt.savefig(f"data/{name}.pdf")

        with open(f"data/{name}.pickle", 'wb') as file:
            pickle.dump(res, file)
        plt.show()
    return res


if __name__ == "__main__":
    problem = run_heat
    sizes = np.arange(MPI.COMM_WORLD.size) + 1
    adaptivity = False

    record_timing(problem, sizes, adaptivity)
