import numpy as np
import scipy
from pySDC.implementations.datatype_classes.mesh import mesh


class SpectralHelper1D:
    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    xp = np

    def __init__(self, N):
        self.N = N

    def get_Id(self):
        raise NotImplementedError

    def get_zero(self):
        return 0 * self.get_Id()

    def get_differentiation_matrix(self):
        raise NotImplementedError()

    def get_integration_matrix(self):
        raise NotImplementedError()

    def get_empty_operator_matrix(self, S, O):
        """
        Return a matrix of operators to be filled with the connections between the solution components.

        Args:
            S (int): Number of components in the solution
            O (sparse matrix): Zero matrix used for initialization

        Returns:
            list containing sparse zeros
        """
        return [[O for _ in range(S)] for _ in range(S)]

    def get_basis_change_matrix(self, *args, **kwargs):
        return self.sparse_lib.eye(self.N)


class ChebychovHelper(SpectralHelper1D):

    def __init__(self, *args, S=1, d=1, mode='T2U', transform_type='fft', **kwargs):
        super().__init__(*args, **kwargs)
        self.S = S
        self.d = d
        self.mode = mode
        self.transform_type = transform_type

        if self.transform_type == 'fft':
            self.get_fft_utils()

        self.cache = {}
        self.norm = self.get_norm()

    def get_1dgrid(self):
        '''
        Generates a 1D grid with Chebychov points. These are clustered at the boundary. You need this kind of grid to
        use discrete cosine transformation (DCT) to get the Chebychov representation. If you want a different grid, you
        need to do an affine transformation before any Chebychov business.

        Returns:
            numpy.ndarray: 1D grid
        '''
        return self.xp.cos(np.pi / self.N * (self.xp.arange(self.N) + 0.5))

    def get_Id(self):
        if self.mode == 'D2U':
            return self.get_conv('T2U') @ self.get_conv('D2T')
        return self.get_conv(self.mode)

    def get_differentiation_matrix(self):
        if self.mode == 'T2T':
            return self.get_T2T_differentiation_matrix()
        elif self.mode == 'T2U':
            return self.get_T2U_differentiation_matrix()
        elif self.mode == 'D2U':
            return self.get_T2U_differentiation_matrix() @ self.get_conv('D2T')
        else:
            raise NotImplementedError(f'{self.mode=!r} not implemented')

    def get_integration_matrix(self):
        if self.mode == 'T2T':
            return self.get_T2T_integration_matrix()
        elif self.mode == 'T2U':
            return self.get_conv('T2U') @ self.get_T2T_integration_matrix()
        else:
            raise NotImplementedError(f'{self.mode=!r} not implemented')

    def get_conv(self, name, N=None):
        '''
        Get conversion matrix between different kinds of polynomials. The supported kinds are
         - T: Chebychov polynomials of first kind
         - U: Chebychov polynomials of second kind
         - D: Dirichlet recombination.

        You get the desired matrix by choosing a name as ``A2B``. I.e. ``T2U`` for the conversion matrix from T to U.
        Once generates matrices are cached. So feel free to call the method as often as you like.

        Args:
         name (str): Conversion code, e.g. 'T2U'
         N (int): Size of the matrix (optional)

        Returns:
            scipy.sparse: Sparse conversion matrix
        '''
        if name in self.cache.keys() and not N:
            return self.cache[name]

        N = N if N else self.N

        def get_forward_conv(name):
            if name == 'T2U':
                mat = (self.sparse_lib.eye(N) - self.sparse_lib.diags(self.xp.ones(N - 2), offsets=+2)) / 2.0
                mat[:, 0] *= 2
            elif name == 'D2T':
                mat = self.sparse_lib.eye(N) - self.sparse_lib.diags(self.xp.ones(N - 2), offsets=+2)
            elif name == 'D2U':
                mat = self.get_conv('D2T') @ self.get_conv('T2U')
            elif name[0] == name[-1]:
                mat = self.sparse_lib.eye(self.N)
            else:
                raise NotImplementedError(f'Don\'t have conversion matrix {name!r}')
            return mat

        try:
            mat = get_forward_conv(name)
        except NotImplementedError as E:
            try:
                fwd = get_forward_conv(name[::-1])
                mat = self.sparse_lib.linalg.inv(fwd.tocsc())
            except NotImplementedError:
                raise E

        self.cache[name] = mat
        return mat

    def get_basis_change_matrix(self, direction='backward'):
        if direction == 'forward':
            return self.get_conv(self.mode)
        else:
            return self.get_conv(self.mode[::-1])

    def get_T2U_differentiation_matrix(self):
        '''
        Sparse differentiation matrix from Chebychov polynomials of first kind to second. When using this, you must
        formulate your problem in first order derivatives.

        Returns:
            scipy.sparse: Sparse differentiation matrix
        '''
        return self.sparse_lib.diags(self.xp.arange(self.N - 1) + 1, offsets=1)

    def get_U2T_integration_matrix(self):
        # TODO: missing integration constant, use T2T instead!
        S = self.sparse_lib.diags(1 / (self.xp.arange(self.N - 1) + 1), offsets=-1).tolil()
        return S

    def get_T2T_integration_matrix(self):
        # TODO: this is a bit fishy
        S = (self.get_U2T_integration_matrix() @ self.get_conv('T2U')).tolil()
        n = self.xp.arange(self.N)
        S[0, 1::2] = (
            (n / (2 * (self.xp.arange(self.N) + 1)))[1::2]
            * (-1) ** (self.xp.arange(self.N // 2))
            / (np.append([1], self.xp.arange(self.N // 2 - 1) + 1))
        )
        return S

    def get_T2T_differentiation_matrix(self, p=1):
        '''
        This is adapted from the Dedalus paper. Keep in mind that the T2T differentiation matrix is dense. But you can
        compute any derivative by simply raising it to some power `p`.

        Args:
            p (int): Derivative you want to compute (can be negative for integration)

        Returns:
            numpy.ndarray: Differentiation matrix
        '''
        D = self.xp.zeros((self.N, self.N))
        for j in range(self.N):
            for k in range(j):
                D[k, j] = 2 * j * ((j - k) % 2)

        D[0, :] /= 2
        return self.sparse_lib.csr_matrix(self.xp.linalg.matrix_power(D, p))

    def get_norm(self):
        '''get normalization for converting Chebychev coefficients and DCT'''
        norm = self.xp.ones(self.N) / self.N
        norm[0] /= 2
        return norm

    def get_fft_utils(self):
        self.fft_utils = {
            'fwd': {},
            'bck': {},
        }

        N = self.N
        k = np.arange(N)
        norm = self.get_norm()

        # forwards transform
        self.fft_utils['fwd']['shuffle'] = np.append(np.arange((N + 1) // 2) * 2, -np.arange(N // 2) * 2 - 1 - N % 2)
        self.fft_utils['fwd']['shift'] = 2 * np.exp(-1j * np.pi * k / (2 * N)) * norm

        # backwards transform
        mask = np.zeros(N, dtype=int)
        mask[: N - N % 2 : 2] = np.arange(N // 2)
        mask[1::2] = N - np.arange(N // 2) - 1
        mask[-1] = N // 2
        self.fft_utils['bck']['shuffle'] = mask

        shift = np.exp(1j * np.pi * k / (2 * N))
        shift[0] /= 2
        self.fft_utils['bck']['shift'] = shift / norm

        return self.fft_utils

    def transform(self, u, axis=-1, **kwargs):
        if self.transform_type == 'dct':
            return self.fft_lib.dct(u, axis=axis) * self.norm
        elif self.transform_type == 'fft':
            result = u.copy()

            shuffle = [slice(0, s, 1) for s in u.shape]
            shuffle[axis] = self.fft_utils['fwd']['shuffle']

            v = u[*shuffle]

            V = self.fft_lib.fft(v, axis=axis, **kwargs)

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            V *= self.fft_utils['fwd']['shift'][*expansion]

            result.real[...] = V.real[...]
            return result
        else:
            raise NotImplementedError

    def itransform(self, u, axis=-1):
        assert self.norm.shape[0] == u.shape[axis]

        if self.transform_type == 'dct':
            return self.fft_lib.idct(u / self.norm, axis=axis)
        elif self.transform_type == 'fft':
            result = u.copy()

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            v = self.fft_lib.ifft(u * self.fft_utils['bck']['shift'][*expansion], axis=axis)

            shuffle = [slice(0, s, 1) for s in u.shape]
            shuffle[axis] = self.fft_utils['bck']['shuffle']
            V = v[*shuffle]

            result.real[...] = V.real[...]
            return result
        else:
            raise NotImplementedError

    def get_Dirichlet_BC_row_T(self, x):
        """
        Get a row for generating Dirichlet BCs at x with T polynomials.
        It returns the values of the T polynomials at x.

        Args:
            x (float): Position of the boundary condition

        Returns:
            self.xp.ndarray: Row to put into a matrix
        """
        if x == -1:
            return (-1) ** self.xp.arange(self.N)
        elif x == 1:
            return self.xp.ones(self.N)
        elif x == 0:
            n = (1 + (-1) ** self.xp.arange(self.N)) / 2
            n[2::4] *= -1
            return n
        else:
            raise NotImplementedError(f'Don\'t know how to generate Dirichlet BC\'s at {x=}!')

    def get_Dirichlet_BC_row_D(self, x):
        res = self.xp.zeros(self.N)
        if x == -1:
            res[0] = 1
            res[1] = -1
        elif x == 1:
            res[0] = 1
            res[1] = 1
        else:
            raise NotImplementedError(f'Don\'t know how to generate Dirichlet BC\'s at {x=}!')
        return res

    # def get_Dirichlet_BC_row_U(self, x):
    #     if x == -1:
    #         n = self.xp.arange(self.N)
    #         return (-1)**n * (n + 1)
    #     elif x == 1:
    #         return self.xp.arange(self.N) + 1
    #     elif x == 0:
    #         return self.get_Dirichlet_BC_row_T(x)
    #         return n
    #     else:
    #         raise NotImplementedError(f'Don\'t know how to generate Dirichlet BC\'s at {x=}!')


class FFTHelper(SpectralHelper1D):

    def __init__(self, *args, x0=0, L=2 * np.pi, **kwargs):
        self.x0 = x0
        self.L = L
        super().__init__(*args, **kwargs)

    def get_1dgrid(self):
        dx = self.L / self.N
        return self.xp.arange(self.N) * dx + self.x0

    def get_wavenumbers(self):
        return self.xp.fft.fftfreq(self.N, 1.0 / self.N)

    def get_differentiation_matrix(self, p=1):
        k = self.get_wavenumbers()
        return self.sparse_lib.linalg.matrix_power(self.sparse_lib.diags(1j * k), p)

    def get_integration_matrix(self, p=1):
        k = self.xp.array(self.get_wavenumbers(), dtype='complex128')
        k[0] = 1j * self.L
        return self.sparse_lib.linalg.matrix_power(self.sparse_lib.diags(1 / (1j * k)), p)

    def get_Id(self):
        return self.sparse_lib.eye(self.N)

    def transform(self, u, axis=-1, **kwargs):
        return self.fft_lib.fft(u, axis=axis, **kwargs)

    def itransform(self, u, axis=-1):
        return self.fft_lib.ifft(u, axis=axis)


class SpectralHelper:
    xp = np
    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    dtype = mesh

    def __init__(self, comm=None):
        self.comm = comm

        self.axes = []
        self.components = []

        self.full_BCs = []
        self.BC_mat = None
        self.BCs = None

        self.fft_cache = {}

    @property
    def u_init(self):
        return self.dtype(self.init)

    @property
    def ndim(self):
        return len(self.axes)

    def add_axis(self, base, *args, **kwargs):

        if base.lower() in ['chebychov', 'chebychev', 'cheby']:
            self.axes.append(ChebychovHelper(*args, **kwargs, transform_type='fft'))
        elif base.lower() in ['fft', 'fourier']:
            self.axes.append(FFTHelper(*args, **kwargs))
        else:
            raise NotImplementedError(f'{base=!r} is not implemented!')

    def add_component(self, name):
        if type(name) in [list, tuple]:
            for me in name:
                self.add_component(me)
        elif type(name) in [str]:
            if name in self.components:
                raise Exception(f'{name=!r} is already added to this problem!')
            self.components.append(name)
        else:
            raise NotImplementedError

    def index(self, name):
        if type(name) in [str, int]:
            return self.components.index(name)
        elif type(name) in [list, tuple]:
            return (self.index(me) for me in name)
        else:
            raise NotImplementedError(f'Don\'t know how to compute index for {type(name)=}')

    def get_empty_operator_matrix(self):
        """
        Return a matrix of operators to be filled with the connections between the solution components.

        Returns:
            list containing sparse zeros
        """
        S = len(self.components)
        O = self.get_Id() * 0
        return [[O for _ in range(S)] for _ in range(S)]

    def get_BC(self, axis, x):
        base = self.axes[axis]
        assert type(base) == ChebychovHelper
        assert x in [-1, 0, 1]

        BC = base.get_Id().tolil() * 0
        BC[-1, :] = base.get_Dirichlet_BC_row_T(x)

        ndim = len(self.axes)
        if ndim == 1:
            return BC
        elif ndim == 2:
            axis2 = (axis + 1) % ndim
            Id = self.get_local_slice_of_1D_matrix(
                self.axes[axis2].get_basis_change_matrix() @ self.axes[axis2].get_Id(), axis=axis2
            )
            mats = [
                None,
            ] * ndim
            mats[axis] = BC
            mats[axis2] = Id
            return self.sparse_lib.kron(*mats)

    def add_BC(self, component, equation, axis, x, v):

        _BC = self.get_BC(axis=axis, x=x)
        self.BC_mat[self.index(equation)][self.index(component)] = _BC
        self.full_BCs += [{'component': component, 'equation': equation, 'axis': axis, 'x': x, 'v': v}]

        shape = self.init[0][1:]
        slices = (
            [self.index(equation)]
            + [slice(0, self.init[0][i + 1]) for i in range(axis)]
            + [-1]
            + [slice(0, self.init[0][i + 1]) for i in range(axis + 1, len(self.axes))]
        )
        # if any(self.BC_rhs_mask[*slices]):
        #     raise Exception('There is already a boundary condition here!')
        self.BC_rhs_mask[*slices] = True
        print(component, equation, axis, slices)

    def setup_BCs(self):
        BC = self.convert_operator_matrix_to_operator(self.BC_mat)

        self.BC_mask = BC != 0
        self._BCs = BC.tolil()[self.BC_mask]
        self.BC_zero_index = self.xp.arange(np.prod(self.init[0]))[self.BC_rhs_mask.flatten()]

    def put_BCs_in_matrix(self, A):
        A = A.tolil()
        # A[self.BC_zero_index, :] = 0  # TODO: do I need sth like this?
        A[self.BC_mask] = self._BCs
        return A.tocsc()

    def put_BCs_in_rhs(self, rhs, istransformed=False):
        assert rhs.ndim > 1, 'rhs must not be flattened here!'

        ndim = len(self.axes)

        for axis in range(ndim):
            if istransformed:
                _rhs_hat = rhs
            else:
                _rhs_hat = self.transform(rhs, axes=(axis - ndim,))
            slices = (
                [slice(0, self.init[0][i + 1]) for i in range(axis)]
                + [-1]
                + [slice(0, self.init[0][i + 1]) for i in range(axis + 1, len(self.axes))]
            )

            for bc in self.full_BCs:
                if axis == bc['axis']:
                    _slice = [self.index(bc['equation'])] + slices
                    _rhs_hat[*_slice] = bc['v']

            if istransformed:
                rhs = _rhs_hat
            else:
                rhs = self.itransform(_rhs_hat, axes=(axis - ndim,))

        return rhs

    def add_equation_lhs(self, A, equation, relations):
        for k, v in relations.items():
            A[self.index(equation)][self.index(k)] = v

    def convert_operator_matrix_to_operator(self, M):
        if len(self.components) == 1:
            return M[0][0]
        else:
            return self.sparse_lib.bmat(M, format='lil')

    def get_grid(self):
        grids = [self.axes[i].get_1dgrid()[self.local_slice[i]] for i in range(len(self.axes))][::-1]
        return self.xp.meshgrid(*grids)

        return self.xp.meshgrid(*[me.get_1dgrid() for me in self.axes[::-1]])

    def get_fft(self, axes, direction):
        shape = self.global_shape[1:]
        key = (axes, direction)

        if key not in self.fft_cache.keys():
            if self.comm is None:
                if direction == 'forward':
                    self.fft_cache[key] = self.xp.fft.fftn
                elif direction == 'backward':
                    self.fft_cache[key] = self.xp.fft.ifftn
                elif direction == 'object':
                    self.fft_cache[key] = None
            else:
                from mpi4py_fft import PFFT, newDistArray

                _fft = PFFT(
                    comm=self.comm,
                    shape=shape,
                    axes=sorted(list(axes)),
                    dtype='D',
                    collapse=False,
                    # backend=self.fft_backend,
                    # comm_backend=self.fft_comm_backend,
                )
                if direction == 'forward':
                    self.fft_cache[key] = _fft.forward
                elif direction == 'backward':
                    self.fft_cache[key] = _fft.backward
                elif direction == 'object':
                    self.fft_cache[key] = _fft

        return self.fft_cache[key]

    def setup_fft(self):
        # if len(self.axes) > 1:
        #     assert all(
        #         type(me) != ChebychovHelper for me in self.axes[:-1]
        #     ), 'Due to handling of imaginary part, we can only have Chebychov in the last dimension!'

        if len(self.components) == 0:
            self.add_component('u')

        shape = [me.N for me in self.axes]
        self.global_shape = (len(self.components),) + tuple(me.N for me in self.axes)
        self.local_slice = [slice(0, me.N) for me in self.axes]

        axes = tuple(i for i in range(len(self.axes)))
        self.fft_obj = self.get_fft(axes=axes, direction='object')
        if self.fft_obj is not None:
            self.local_slice = self.fft_obj.local_slice(False)

        self.init = (np.empty(shape=self.global_shape)[:, *self.local_slice].shape, self.comm, np.dtype('complex128'))

        self.BC_mat = self.get_empty_operator_matrix()
        self.BC_mask = self.xp.zeros(
            shape=self.init[0],
            dtype=bool,
        )
        self.BC_rhs_mask = self.xp.zeros(
            shape=self.init[0],
            dtype=bool,
        )

    def _redistribute(self, u, axis):
        if self.comm is None:
            return u
        else:
            return u.redistribute(axis)

    def _transform_fft(self, u, axes):
        fft = self.get_fft(axes, 'forward')
        return fft(u, axes=axes)

    def _transform_dct(self, u, axes):
        result = u.copy()

        if len(axes) > 1:
            v = self._transform_dct(self._transform_dct(u, axes[1:]), (axes[0],))
        else:

            v = u.copy()
            axis = axes[0]

            shuffle = [slice(0, s, 1) for s in u.shape]
            shuffle[axis] = self.axes[axis].fft_utils['fwd']['shuffle']
            v = v[*shuffle]

            fft = self.get_fft(axes, 'forward')
            v = fft(v, axes=axes)

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, v.shape[axis], 1)
            v *= self.axes[axis].fft_utils['fwd']['shift'][*expansion]

        result.real[...] = v.real
        return result

    def transform(self, u, axes=None):
        trfs = {
            ChebychovHelper: self._transform_dct,
            FFTHelper: self._transform_fft,
        }

        axes = tuple(-i - 1 for i in range(self.ndim)) if axes is None else axes

        result = u.copy().astype(complex)

        for comp in self.components:
            i = self.index(comp)

            axes_base = []
            for base in trfs.keys():
                axes_base = tuple(me for me in axes if type(self.axes[me]) == base)

                if len(axes_base) > 0:
                    fft = self.get_fft(axes_base, 'object')
                    if fft:
                        from mpi4py_fft import newDistArray

                        _out = newDistArray(fft, True)
                        _in = newDistArray(fft, False).redistribute(-1)
                        _in[...] = result[i]
                        _in = _in.redistribute(axes_base[-1])
                    else:
                        _in = result[i]
                        _out = _in

                    _out[...] = trfs[base](_in, axes=axes_base)

                    if fft:
                        result[i] = _out.redistribute(-1) * np.prod([self.axes[i].N for i in axes_base])
                    else:
                        result[i] = _out

        return result

    def _transform_ifft(self, u, axes):
        ifft = self.get_fft(axes, 'backward')
        return ifft(u, axes=axes)

    def _transform_idct(self, u, axes):
        result = u.copy()

        v = u.copy().astype(complex)

        if len(axes) > 1:
            v = self._transform_idct(self._transform_idct(u, axes[1:]), (axes[0],))
        else:
            axis = axes[0]

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            v *= self.axes[axis].fft_utils['bck']['shift'][*expansion]

            ifft = self.get_fft(axes, 'backward')
            v = ifft(v, axes=axes)

            shuffle = [slice(0, s, 1) for s in u.shape]
            shuffle[axis] = self.axes[axis].fft_utils['bck']['shuffle']
            v = v[*shuffle]

        result.real[...] = v.real
        return result

    def itransform(self, u, axes=None):
        trfs = {
            FFTHelper: self._transform_ifft,
            ChebychovHelper: self._transform_idct,
        }

        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes

        result = u.copy().astype(complex)

        for comp in self.components:
            i = self.index(comp)

            axes_base = []
            for base in trfs.keys():
                axes_base = tuple(me for me in axes if type(self.axes[me]) == base)

                if len(axes_base) > 0:
                    fft = self.get_fft(axes_base, 'object')
                    if fft:
                        from mpi4py_fft import newDistArray

                        _out = newDistArray(fft, False)
                        _in = newDistArray(fft, True).redistribute(-1)
                        _in[...] = result[i]
                        _in = _in.redistribute(axes_base[0]) / np.prod([self.axes[i].N for i in axes_base])
                    else:
                        _in = result[i]
                        _out = _in

                    _out[...] = trfs[base](_in, axes=axes_base)

                    if fft:
                        result[i] = _out.redistribute(-1)
                    else:
                        result[i] = _out

        return result

    def get_local_slice_of_1D_matrix(self, M, axis):
        return M.tolil()[self.local_slice[axis], self.local_slice[axis]]

    def get_differentiation_matrix(self, axes, **kwargs):
        """
        Get differentiation matrix along specified axis.

        Args:
            axes (tuple): Axes along which to differentiate.

        Returns:
            sparse differentiation matrix
        """
        sp = self.sparse_lib
        D = sp.eye(np.prod(self.init[0][1:]), dtype=complex).tolil() * 0
        ndim = len(self.axes)

        if ndim == 1:
            D = self.axes[0].get_differentiation_matrix(**kwargs)
        elif ndim == 2:
            for axis in axes:
                axis2 = (axis + 1) % ndim
                D1D = self.axes[axis].get_differentiation_matrix(**kwargs)

                if len(axes) > 1:
                    I1D = sp.eye(self.axes[axis2].N)
                else:
                    I1D = self.axes[axis2].get_Id()

                mats = [None] * ndim
                mats[axis] = self.get_local_slice_of_1D_matrix(D1D, axis)
                mats[axis2] = self.get_local_slice_of_1D_matrix(I1D, axis2)

                if axis == axes[0]:
                    D += sp.kron(*mats)
                else:
                    D = D @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Differentiation matrix not implemented for {ndim} dimension!')

        return D

    def get_integration_matrix(self, axes):
        """
        Get integration matrix to integrate along specified axis.

        Args:
            axes (tuple): Axes along which to integrate over.

        Returns:
            sparse integration matrix
        """
        sp = self.sparse_lib
        S = sp.eye(np.prod(self.init[0][1:]), dtype=complex).tolil() * 0
        ndim = len(self.axes)

        if ndim == 1:
            S = self.axes[0].get_integration_matrix()
        elif ndim == 2:
            for axis in axes:
                axis2 = (axis + 1) % ndim
                S1D = self.axes[axis].get_integration_matrix()

                if len(axes) > 1:
                    I1D = sp.eye(self.axes[axis2].N)
                else:
                    I1D = self.axes[axis2].get_Id()

                mats = [None] * ndim
                mats[axis] = self.get_local_slice_of_1D_matrix(S1D, axis)
                mats[axis2] = self.get_local_slice_of_1D_matrix(I1D, axis2)

                if axis == axes[0]:
                    S += sp.kron(*mats)
                else:
                    S = S @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Integration matrix not implemented for {ndim} dimension!')

        return S

    def get_Id(self):
        """
        Get identity matrix

        Returns:
            sparse identity matrix
        """
        sp = self.sparse_lib
        ndim = self.ndim
        I = sp.eye(np.prod(self.init[0][1:]), dtype=complex).tolil()

        if ndim == 1:
            I = self.axes[0].get_Id()
        elif ndim == 2:
            for axis in range(ndim):
                axis2 = (axis + 1) % ndim
                I1D = self.axes[axis].get_Id()

                I1D2 = sp.eye(self.axes[axis2].N)

                mats = [None] * ndim
                mats[axis] = self.get_local_slice_of_1D_matrix(I1D, axis)
                mats[axis2] = self.get_local_slice_of_1D_matrix(I1D2, axis2)

                I = I @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Identity matrix not implemented for {ndim} dimension!')

        return I

    def get_basis_change_matrix(self, axes=None, direction='backward'):
        """
        Some spectral bases do a change between bases while differentiating. You can use this matrix to change back to
        the original basis afterwards.

        Args:
            axes (tuple): Axes along which to change basis.
            direction (str): Direction of the basis change

        Returns:
            sparse basis change matrix
        """
        axes = tuple(-i - 1 for i in range(self.ndim)) if axes is None else axes

        sp = self.sparse_lib
        C = sp.eye(np.prod(self.init[0][1:]), dtype=complex).tolil() * 0
        ndim = len(self.axes)

        if ndim == 1:
            C = self.axes[0].get_basis_change_matrix(direction=direction)
        elif ndim == 2:
            for axis in axes:
                axis2 = (axis + 1) % ndim
                C1D = self.axes[axis].get_basis_change_matrix(direction=direction)

                if len(axes) > 1:
                    I1D = sp.eye(self.axes[axis2].N)
                else:
                    I1D = self.axes[axis2].get_Id()

                mats = [None] * ndim
                mats[axis] = self.get_local_slice_of_1D_matrix(C1D, axis)
                mats[axis2] = self.get_local_slice_of_1D_matrix(I1D, axis2)

                if axis == axes[0]:
                    C += sp.kron(*mats)
                else:
                    C = C @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Integration matrix not implemented for {ndim} dimension!')

        return C
