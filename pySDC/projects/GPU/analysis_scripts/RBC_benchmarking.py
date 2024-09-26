import time
import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_problem(nz, useGPU):
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    return RayleighBenard(nx=nz * 4, nz=nz, useGPU=useGPU)


def time_eval_f(prob):
    u = prob.u_exact(0)

    t0 = time.perf_counter()

    prob.eval_f(u)

    prob.comm.Barrier()

    t1 = time.perf_counter()

    elapsed_time = t1 - t0
    return elapsed_time


def time_solve_system(prob, use_cache=False):
    dt = 1.0
    u = prob.u_exact(0)

    if not use_cache:
        prob.cached_factorizations = {}
    else:
        prob.solve_system(u, dt)

    t0 = time.perf_counter()

    prob.solve_system(u, dt)

    prob.comm.Barrier()

    t1 = time.perf_counter()

    elapsed_time = t1 - t0
    return elapsed_time


def time_solve_system_with_cache(prob):
    return time_solve_system(prob, use_cache=True)


def multiple_measurements(function, N, nz, useGPU, run=True, num_procs=None):
    data = []
    import pySDC.projects.GPU as project

    problem = None

    if num_procs is None:
        problem = get_problem(nz=nz, useGPU=useGPU)
        num_procs = problem.comm.size

    path = f'{project.__file__[:-12]}/data/{function.__name__}-{nz}-{useGPU}-{num_procs}.pickle'

    if run:
        if problem is None:
            problem = get_problem(nz=nz, useGPU=useGPU)

        for _ in range(N):
            data += [function(problem)]

        with open(path, 'wb') as file:
            pickle.dump(data, file)
    else:
        with open(path, 'rb') as file:
            data = pickle.load(file)

    return data


if __name__ == '__main__':
    run = False
    useGPUs = [False, True]
    num_procss = [16, 4]
    N = 4
    nzs = [16, 32, 64, 128, 256]
    functions = [time_eval_f, time_solve_system, time_solve_system_with_cache]

    timings = {}

    for useGPU, num_procs in zip(useGPUs, num_procss):
        timings[useGPU] = {}

        for function in functions:
            timings[useGPU][function.__name__] = []

            for nz in nzs:
                kwargs = {'function': function, 'N': N, 'nz': nz, 'useGPU': useGPU, 'run': run, 'num_procs': num_procs}

                data = multiple_measurements(**kwargs)
                timings[useGPU][function.__name__] += [np.mean(data)]

    colors = {
        'time_eval_f': 'teal',
        'time_solve_system': 'orange',
        'time_solve_system_with_cache': 'magenta',
    }
    fig, ax = plt.subplots()
    for useGPU in useGPUs:
        gpu_label = 'GPU' if useGPU else 'CPU'
        ls = '-' if useGPU else '--'
        for function in functions:
            ax.loglog(
                nzs,
                timings[useGPU][function.__name__],
                label=f"{function.__name__[4:].replace('_', ' ')} {gpu_label}",
                color=colors[function.__name__],
                ls=ls,
            )

    ax.legend(frameon=False)
    ax.set_xlabel('$N_z$')
    ax.set_ylabel('$t$')
    plt.show()
