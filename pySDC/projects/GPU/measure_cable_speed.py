import cupy as xp
from cupy.cuda import nccl
from cupyx.profiler import benchmark
from mpi4py import MPI
import pickle


def get_path(name, shape):
    return f'out/cable_speed_{name}_{shape[0]}x{shape[1]}.pickle'


def get_communicators():
    comm = MPI.COMM_WORLD
    uid = comm.bcast(nccl.get_unique_id(), root=0)
    commNCCL = nccl.NcclCommunicator(comm.size, uid, comm.rank)
    return comm, commNCCL


def measure_speed(comm, commNCCL, shape, name='', **kwargs):
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
        results[mode] = {'CPU': res.cpu_times.mean(), 'GPU': res.gpu_times.mean(), 'size': size}
        assert xp.linalg.norm(data, 2.0) > 0

    print(results)

    with open(get_path(name=name, shape=shape), 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    comm, commNCCL = get_communicators()
    for power in [22, 24, 26]:
        measure_speed(
            comm,
            commNCCL,
            (
                2**power,
                2**3,
            ),
            n_repeat=1000,
        )
