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


def multiple_measurements(function, N, nz, useGPU, run=True):
    data = []
    import pySDC.projects.GPU as project

    problem = get_problem(nz=nz, useGPU=useGPU)

    path = f'{project.__file__[:-12]}/data/{function.__name__}-{nz}-{useGPU}-{problem.comm.size}.pickle'

    if run:

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
    useGPUs = [False]
    N = 4
    nzs = [32, 64, 128]
    functions = [time_eval_f, time_solve_system, time_solve_system_with_cache]

    timings = {}

    for useGPU in useGPUs:
        timings[useGPU] = {}

        for function in functions:
            timings[useGPU][function.__name__] = []

            for nz in nzs:
                kwargs = {'function': function, 'N': N, 'nz': nz, 'useGPU': useGPU, 'run': run}

                data = multiple_measurements(**kwargs)
                timings[useGPU][function.__name__] += [np.mean(data)]
    print(timings)

    fig, ax = plt.subplots()
    for function in functions:
        ax.loglog(nzs, timings[useGPU][function.__name__], label=function.__name__[4:].replace('_', ' '))

    ax.legend(frameon=False)
    plt.show()
