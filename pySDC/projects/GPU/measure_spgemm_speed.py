from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D
from time import perf_counter
import pickle
import matplotlib.pyplot as plt
import numpy as np


def get_mats(N, useGPU, types):
    prob = RayleighBenard3D(nx=N, ny=N, nz=N, left_preconditioner=True, Dirichlet_recombination=True)

    mats = [prob.Pl, prob.L]

    if useGPU:
        import cupyx.scipy.sparse as sp
    else:
        import scipy.sparse as sp

    for i in range(len(mats)):
        if types[i] == 'csc':
            mats[i] = sp.csc_matrix(mats[i])
        elif types[i] == 'csr':
            mats[i] = sp.csr_matrix(mats[i])
        else:
            raise NotImplementedError

    return mats


def single_experiment(mats, useGPU):
    if useGPU:
        import cupy as cp

        t0 = cp.cuda.Event()
        t1 = cp.cuda.Event()
        t0.record()
    else:
        t0 = perf_counter()

    mats[0] @ mats[1]

    if useGPU:
        t1.record()
        t1.synchronize()
        return cp.cuda.get_elapsed_time(t0, t1) / 1e3
    else:
        t1 = perf_counter()
        return t1 - t0


def sample_experiment(mats, useGPU, num_experiments=20):
    timings = np.zeros(num_experiments)
    for i in range(num_experiments):
        timings[i] = single_experiment(mats, useGPU)
    return timings


def get_path(N, useGPU, types):
    return f'./data/spgemm_timings_N{N}_useGPU{useGPU}_{types[0]}x{types[1]}.pickle'


def get_label(useGPU, types=None):
    label = 'GPU' if useGPU else 'CPU'
    if types is not None:
        label += f' {types[0].upper()}x{types[1].upper()}'
    return label


def record_experiment(N, useGPU, types, **kwargs):
    mats = get_mats(N, useGPU, types)
    path = get_path(N, useGPU, types)

    _ = sample_experiment(mats, useGPU, num_experiments=2)
    timings = sample_experiment(mats, useGPU, **kwargs)

    with open(path, 'wb') as file:
        pickle.dump(timings, file)
    print(f'Recorded timings at {path}', flush=True)


def record_experiments(Ns, **kwargs):
    for N in Ns:
        record_experiment(N=N, **kwargs)


def plot_timings(Ns, ax, **kwargs):
    timings = {}
    for N in Ns:
        with open(get_path(N=N, **kwargs), 'rb') as file:
            timings[N] = pickle.load(file)

    mean_time = [np.mean(timings[N]) for N in Ns]
    ax.loglog([5 * me**3 for me in Ns], mean_time, label=get_label(**kwargs))
    ax.legend(frameon=False)
    ax.set_xlabel('$N$')
    ax.set_ylabel('$t$ / s')


def compare_csr_csc(useGPU=False, record=False):
    Ns = [2**me for me in range(1, 8)]
    Ns = range(16, 100, 4)
    types = [['csr', 'csc'], ['csc', 'csc'], ['csc', 'csr'], ['csr', 'csr']]
    types = [['csr', 'csr'], ['csc', 'csc']]
    fig, ax = plt.subplots()

    for _types in types:
        if record:
            record_experiments(Ns, useGPU=useGPU, types=_types)
        plot_timings(Ns, ax=ax, useGPU=useGPU, types=_types)
    figpath = f'plots/SPGEMM_csr_vs_csc_{get_label(useGPU)}.pdf'
    plt.savefig(figpath, bbox_inches='tight')
    print(f'Plotted {figpath}', flush=True)
    plt.show()


def compare_CPU_GPU(record=False):
    types = [['csc', 'csc']]
    Ns = range(16, 100, 4)
    fig, ax = plt.subplots()
    for useGPU in [True, False]:
        for _types in types:
            if record:
                record_experiments(Ns, useGPU=useGPU, types=_types)
            plot_timings(Ns, ax=ax, useGPU=useGPU, types=_types)
    figpath = f'plots/SPGEMM_CPU_vs_GPU_{get_label(useGPU)}.pdf'
    plt.savefig(figpath, bbox_inches='tight')
    print(f'Plotted {figpath}', flush=True)


if __name__ == '__main__':
    # for useGPU in [True, False]:
    #     compare_csr_csc(useGPU, record=False)
    compare_CPU_GPU(False)
