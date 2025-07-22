try:
    import mpi4py

    MPI_avail = True
except ModuleNotFoundError:
    MPI_avail = False


try:
    import cupy

    GPU_avail = True
except ModuleNotFoundError:
    GPU_avail = False
