import cupy as cp
from pySDC.core.Errors import DataError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class cupy_mesh(cp.ndarray):
    """
    CuPy-based datatype for serial or parallel meshes.
    """

    def __new__(cls, init, val=0.0, offset=0, buffer=None, strides=None, order=None):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        if isinstance(init, cupy_mesh):
            obj = cp.ndarray.__new__(cls, shape=init.shape, dtype=init.dtype, strides=strides, order=order)
            obj[:] = init[:]
            obj._comm = init._comm
        elif (
            isinstance(init, tuple)
            and (init[1] is None or isinstance(init[1], MPI.Intracomm))
            and isinstance(init[2], cp.dtype)
        ):
            obj = cp.ndarray.__new__(cls, init[0], dtype=init[2], strides=strides, order=order)
            obj.fill(val)
            obj._comm = init[1]
        else:
            raise NotImplementedError(type(init))
        return obj

    @property
    def comm(self):
        """
        Getter for the communicator
        """
        return self._comm

    @property
    def xp(self):
        '''
        Use this to infer what numerical library to use with this data.
        '''
        return cp

    def __array_finalize__(self, obj):
        """
        Finalizing the datatype. Without this, new datatypes do not 'inherit' the communicator.
        """
        if obj is None:
            return
        self._comm = getattr(obj, '_comm', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Overriding default ufunc, cf. https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs
        """
        args = []
        comm = None
        for _, input_ in enumerate(inputs):
            if isinstance(input_, cupy_mesh):
                args.append(input_.view(cp.ndarray))
                comm = input_.comm
            else:
                args.append(input_)
        results = super(cupy_mesh, self).__array_ufunc__(ufunc, method, *args, **kwargs).view(cupy_mesh)
        if not method == 'reduce':
            results._comm = comm
        return results

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = float(cp.amax(cp.ndarray.__abs__(self)))

        if self.comm is not None:
            if self.comm.Get_size() > 1:
                global_absval = 0.0
                global_absval = max(self.comm.allreduce(sendobj=local_absval, op=MPI.MAX), global_absval)
            else:
                global_absval = local_absval
        else:
            global_absval = local_absval

        return float(global_absval)

    def isend(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (non-blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            request handle
        """
        return comm.Issend(self[:], dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        return comm.Irecv(self[:], source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """
        comm.Bcast(self[:], root=root)
        return self


class MultiComponentMeshCuPy(cupy_mesh):
    r"""
    See documentation of `MultiComponentMesh`, which is also valid for this class.
    """

    components = []

    def __new__(cls, init, *args, **kwargs):
        if isinstance(init, tuple):
            shape = (init[0],) if type(init[0]) is int else init[0]
            obj = super().__new__(cls, ((len(cls.components), *shape), *init[1:]), *args, **kwargs)
        else:
            obj = super().__new__(cls, init, *args, **kwargs)

        return obj

    def __getattr__(self, name):
        if name in self.components:
            if self.shape[0] == len(self.components):
                return self[self.components.index(name)]
            else:
                raise AttributeError(f'Cannot access {name!r} in {type(self)!r} because the shape is unexpected.')
        else:
            raise AttributeError(f"{type(self)!r} does not have attribute {name!r}!")


class imex_cupy_mesh(MultiComponentMeshCuPy):
    components = ['impl', 'expl']
