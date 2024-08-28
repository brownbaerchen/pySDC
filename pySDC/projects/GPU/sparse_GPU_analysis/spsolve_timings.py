import scipy.sparse.linalg as lina
import scipy.sparse as sp
import numpy as xp
import pickle


def gen_data(sp, xp, N):
    A = sp.eye(N).tocsr()
    y = xp.zeros(N)
    return A, y


def single_test_dense(A, y, xp):
    xp.linalg.solve(A, y)
    # lina.gmres(A, y)
    # lina.cg(A, y)


def single_test(A, y, lina):
    lina.spsolve(A, y)
    # lina.gmres(A, y)
    # lina.cg(A, y)


from cupyx.profiler import benchmark


def test_CPU(N):
    import numpy as xp
    import scipy.sparse as sp
    import scipy.sparse.linalg as lina

    A, y = gen_data(sp, xp, N)
    args = (A, y, lina)
    times = benchmark(single_test, args, n_repeat=250)
    with open(f'data/CPU_spsolve_N{N}.pickle', 'wb') as file:
        data = {'CPU': times.cpu_times, 'GPU': times.gpu_times}
        pickle.dump(times, file)


def test_GPU(N):
    import cupy as xp
    import cupyx.scipy.sparse as sp
    import cupyx.scipy.sparse.linalg as lina

    A, y = gen_data(sp, xp, N)
    args = (A, y, lina)
    times = benchmark(single_test, args, n_repeat=250)
    with open(f'data/GPU_spsolve_N{N}.pickle', 'wb') as file:
        data = {'CPU': times.cpu_times, 'GPU': times.gpu_times}
        pickle.dump(times, file)

    # A, y = gen_data(sp, xp, N)
    # args = (A.toarray(), y, xp)
    # print(benchmark(single_test_dense, args, n_repeat=250))


def plot(Ns):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    t_cpu = []
    t_gpu = []

    for N in Ns:
        with open(f'data/CPU_spsolve_N{N}.pickle', 'rb') as file:
            data_CPU = pickle.load(file)
        with open(f'data/GPU_spsolve_N{N}.pickle', 'rb') as file:
            data_GPU = pickle.load(file)

        t_cpu += [max([data_CPU.cpu_times.mean(), data_CPU.gpu_times.mean()])]
        t_gpu += [max([data_GPU.cpu_times.mean(), data_GPU.gpu_times.mean()])]

    ax.plot(Ns, t_cpu, label='CPU')
    ax.plot(Ns, t_gpu, label='GPU')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$t$')
    ax.set_title('Performance of spsolve')
    fig.savefig('spsolve_timings.pdf', bbox_inches='tight')


if __name__ == '__main__':
    exps = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    Ns = [2**e for e in exps]
    for N in Ns:
        test_CPU(N=N)
        test_GPU(N=N)
    plot(Ns)
