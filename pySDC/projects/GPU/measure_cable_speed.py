from mpi4py import MPI
import pickle


powers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
shapes = [
    (
        2**power,
        2**3,
    )
    for power in powers
]


def get_path(name, shape):
    return f'out/cable_speed_{name}_{shape[0]}x{shape[1]}.pickle'


def get_communicators():
    from cupy.cuda import nccl

    comm = MPI.COMM_WORLD
    uid = comm.bcast(nccl.get_unique_id(), root=0)
    commNCCL = nccl.NcclCommunicator(comm.size, uid, comm.rank)
    return comm, commNCCL


def measure_speed(comm, commNCCL, shape, name='', **kwargs):
    import cupy as xp
    from cupy.cuda import nccl
    from cupyx.profiler import benchmark

    data = xp.random.random(shape, dtype='d')
    size = data.nbytes
    results = {}

    def speed_MPI(_data, group=None):
        if comm.rank == 0:
            comm.Send(data, dest=1)
        if comm.rank == 1:
            comm.Recv(data, source=0)

    def speed_NCCL(_data, group=False):
        stream = xp.cuda.get_current_stream().ptr
        if group:
            nccl.groupStart()
        if comm.rank == 0:
            commNCCL.send(data.data.ptr, data.size, nccl.NCCL_FLOAT64, 1, stream)
        if comm.rank == 1:
            commNCCL.recv(data.data.ptr, data.size, nccl.NCCL_FLOAT64, 0, stream)
        if group:
            nccl.groupEnd()

    for mode, func in zip(['MPI', 'NCCL', 'NCCL group'], [speed_MPI, speed_NCCL, speed_NCCL]):
        if comm.rank > 0:
            data[:] = 0

        args = (data,)
        if mode == 'NCCL group':
            args += (True,)

        res = benchmark(func, args, **kwargs)
        results[mode] = {
            'CPU_time': res.cpu_times.mean(),
            'CPU_std': res.cpu_times.std(),
            'GPU_time': res.gpu_times.mean(),
            'GPU_std': res.gpu_times.std(),
            'size': size,
        }
        assert xp.linalg.norm(data, 2.0) > 0

    print(results)

    with open(get_path(name=name, shape=shape), 'wb') as file:
        pickle.dump(results, file)


def record(shapes):
    comm, commNCCL = get_communicators()
    for shape in shapes:
        measure_speed(comm, commNCCL, shape, n_repeat=10000, n_warmup=20)


def plot(shapes, name=''):
    from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl
    import matplotlib.pyplot as plt

    setup_mpl()
    figsize = figsize_by_journal('JSC_thesis', 1.0, 0.4)

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

    results = []
    for shape in shapes:
        with open(get_path(name=name, shape=shape), 'rb') as file:
            data = pickle.load(file)
        results += [data]

    colors = {
        'MPI': 'tab:blue',
        'NCCL': 'tab:orange',
        'NCCL group': 'tab:green',
    }
    ls = {
        'MPI': '-',
        'NCCL': '--',
        'NCCL group': ':',
    }

    for lib in ['NCCL', 'MPI']:
        axs[0].errorbar(
            [me[lib]['size'] for me in results],
            [me[lib]['GPU_time'] for me in results],
            yerr=[me[lib]['GPU_std'] for me in results],
            label=lib,
            color=colors[lib],
            ls=ls[lib],
        )
        axs[1].errorbar(
            [me[lib]['size'] for me in results],
            [me[lib]['size'] / me[lib]['GPU_time'] for me in results],
            yerr=[me[lib]['size'] * me[lib]['GPU_std'] / me[lib]['GPU_time'] ** 2 for me in results],
            label=lib,
            color=colors[lib],
            ls=ls[lib],
        )

    axs[1].axhline(200 / 8 * 1e9, label='Infiniband limit', color='black')

    axs[0].legend(frameon=False)
    axs[0].set_xlabel('size / B')
    axs[0].set_ylabel('time / s')
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')

    axs[1].legend(frameon=False)
    axs[1].set_ylabel('bandwidth / B/s')
    axs[1].set_xlabel('size / B')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    fig.tight_layout()

    fig.savefig('plots/p2p_cable_speed.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot(shapes)
