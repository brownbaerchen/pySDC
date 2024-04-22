from mpi4py import MPI
import pickle


powers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
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

    def speed_MPI(_data):
        if comm.rank == 0:
            comm.Send(data, dest=1)
        if comm.rank == 1:
            comm.Recv(data, source=0)

    def speed_NCCL(_data):
        stream = xp.cuda.get_current_stream().ptr
        nccl.groupStart()
        if comm.rank == 0:
            commNCCL.send(data.data.ptr, data.size, nccl.NCCL_FLOAT64, 1, stream)
        if comm.rank == 1:
            commNCCL.recv(data.data.ptr, data.size, nccl.NCCL_FLOAT64, 0, stream)
        nccl.groupEnd()

    for mode, func in zip(['MPI', 'NCCL'], [speed_MPI, speed_NCCL]):
        if comm.rank > 0:
            data[:] = 0

        res = benchmark(func, (data,), **kwargs)
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
        measure_speed(comm, commNCCL, shape, n_repeat=1000)


def plot(shapes, name=''):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharex=True)

    results = []
    for shape in shapes:
        with open(get_path(name=name, shape=shape), 'rb') as file:
            data = pickle.load(file)
        results += [data]

    colors = {
        'MPI': 'blue',
        'NCCL': 'red',
    }

    for lib in ['NCCL', 'MPI']:
        axs[0].errorbar(
            [me[lib]['size'] for me in results],
            [me[lib]['GPU_time'] for me in results],
            yerr=[me[lib]['GPU_std'] for me in results],
            label=lib,
            color=colors[lib],
        )
        axs[1].errorbar(
            [me[lib]['size'] for me in results],
            [me[lib]['size'] / me[lib]['GPU_time'] for me in results],
            yerr=[me[lib]['size'] * me[lib]['GPU_std'] / me[lib]['GPU_time'] ** 2 for me in results],
            label=lib,
            color=colors[lib],
        )

    axs[1].axhline(200 / 8 * 1e9, label='Infiniband limit', color='black')

    axs[0].legend(frameon=False)
    axs[0].set_xlabel('size / byte')
    axs[0].set_ylabel('time / s')
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')

    axs[1].legend(frameon=False)
    axs[1].set_ylabel('bandwidth / byte/s')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    record(shapes)
