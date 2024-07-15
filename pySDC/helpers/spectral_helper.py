import numpy as np
import scipy


class SpectralHelperBase:
    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    xp = np

    def __init__(self, N, sparse_format='lil'):
        self.N = N
        self.sparse_format = sparse_format

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

    def get_basis_change_matrix(self):
        return self.sparse_lib.eye(self.N)


class ChebychovHelper(SpectralHelperBase):

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
                mat = (
                    self.sparse_lib.eye(N, format=self.sparse_format)
                    - self.sparse_lib.diags(self.xp.ones(N - 2), offsets=+2, format=self.sparse_format)
                ) / 2.0
                mat[:, 0] *= 2
            elif name == 'D2T':
                mat = self.sparse_lib.eye(N, format=self.sparse_format) - self.sparse_lib.diags(
                    self.xp.ones(N - 2), offsets=+2, format=self.sparse_format
                )
            elif name == 'D2U':
                mat = self.get_conv('D2T') @ self.get_conv('T2U')
            elif name[0] == name[-1]:
                mat = self.sparse_lib.eye(self.N, format=self.sparse_format)
            else:
                raise NotImplementedError(f'Don\'t have conversion matrix {name!r}')
            return mat

        try:
            mat = get_forward_conv(name)
        except NotImplementedError as E:
            try:
                fwd = get_forward_conv(name[::-1])
                mat = self.sparse_lib.linalg.inv(fwd)
            except NotImplementedError:
                raise E

        self.cache[name] = mat
        return mat

    def get_basis_change_matrix(self):
        return self.get_conv(self.mode[::-1])

    def get_T2U_differentiation_matrix(self):
        '''
        Sparse differentiation matrix from Chebychov polynomials of first kind to second. When using this, you must
        formulate your problem in first order derivatives.

        Returns:
            scipy.sparse: Sparse differentiation matrix
        '''
        return self.sparse_lib.diags(self.xp.arange(self.N - 1) + 1, offsets=1, format=self.sparse_format)

    def get_U2T_integration_matrix(self):
        # TODO: missing integration constant, use T2T instead!
        S = self.sparse_lib.diags(1 / (self.xp.arange(self.N - 1) + 1), offsets=-1, format=self.sparse_format).tolil()
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
        return self.xp.linalg.matrix_power(D, p)

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
        # self.fft_utils['fwd']['shift'][0] += 2j

        # backwards transform
        mask = np.zeros(N, dtype=int)
        mask[: N - N % 2 : 2] = np.arange(N // 2)
        mask[1::2] = N - np.arange(N // 2) - 1
        mask[-1] = N // 2
        self.fft_utils['bck']['shuffle'] = mask

        shift = np.exp(1j * np.pi * k / (2 * N))
        shift[0] /= 2
        self.fft_utils['bck']['shift'] = shift / norm

        # self.fft_utils['fwd']['shift'][...] = 1
        # self.fft_utils['fwd']['shuffle'] = slice(0, N, 1)

        return self.fft_utils

    def transform(self, u, axis=-1, **kwargs):
        if self.transform_type == 'dct':
            return self.fft_lib.dct(u, axis=axis) * self.norm
        elif self.transform_type == 'fft':
            result = u.copy()

            slices = [slice(0, s, 1) for s in u.shape]
            slices[axis] = self.fft_utils['fwd']['shuffle']

            v = u[*slices]

            V = self.fft_lib.fft(v, axis=axis, **kwargs)

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            V *= self.fft_utils['fwd']['shift'][*expansion]

            # # prune complex part
            # expansion1 = [slice(0, s, 1) for s in u.shape]
            # expansion1[axis] = slice(1, self.N, 1)
            # expansion2 = [slice(0, s, 1) for s in u.shape]
            # expansion2[axis] = slice(self.N, 0, -1)
            # V[*expansion1] += V.real[*expansion2]*1j
            result.real[...] = V.real[...]
            return result
        else:
            raise NotImplementedError

    def itransform(self, u, axis=-1):
        assert self.norm.shape[0] == u.shape[axis]

        if self.transform_type == 'dct':
            return self.fft_lib.idct(u / self.norm, axis=axis)
            # return self.fft_lib.idct(u / self.norm.reshape(u.shape), axis=axis)
        elif self.transform_type == 'fft':
            result = u.copy()

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            v = self.fft_lib.ifft(u * self.fft_utils['bck']['shift'][*expansion], axis=axis)

            slices = [slice(0, s, 1) for s in u.shape]
            slices[axis] = self.fft_utils['bck']['shuffle']
            V = v[*slices]

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


class FFTHelper(SpectralHelperBase):

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
        return self.sparse_lib.eye(self.N, format=self.sparse_format)

    def transform(self, u, axis=-1, **kwargs):
        return self.fft_lib.fft(u, axis=axis, **kwargs)

    def itransform(self, u, axis=-1):
        return self.fft_lib.ifft(u, axis=axis)

    def get_fft_utils(self):
        """
        No need to do anything. We just want a common interface with the dct of Chebychov.
        """
        self.fft_utils = {
            'fwd': {},
            'bck': {},
        }

        N = self.N

        # forwards transform
        self.fft_utils['fwd']['shuffle'] = slice(0, N, 1)
        self.fft_utils['fwd']['shift'] = self.xp.ones(N)

        # backwards transform
        self.fft_utils['bck']['shuffle'] = slice(0, N, 1)
        self.fft_utils['bck']['shift'] = self.xp.ones(N)

        return self.fft_utils


class SpectralHelper:
    xp = np
    fft_lib = scipy.fft
    sparse_lib = scipy.sparse

    def __init__(self):
        self.axes = []
        self.components = []
        self.BC_mat = None
        self.BCs = None

    def add_axis(self, base, *args, **kwargs):

        if base.lower() in ['chebychov', 'chebychev', 'cheby']:
            self.axes.append(ChebychovHelper(*args, **kwargs, transform_type='fft'))
        elif base.lower() in ['fft', 'fourier']:
            self.axes.append(FFTHelper(*args, **kwargs))
        else:
            raise NotImplementedError(f'{base=!r} is not implemented!')

        self.axes[-1].get_fft_utils()

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
        return self.components.index(name)

    def get_empty_operator_matrix(self):
        """
        Return a matrix of operators to be filled with the connections between the solution components.

        Returns:
            list containing sparse zeros
        """
        S = len(self.components)
        O = self.get_Id() * 0
        return [[O for _ in range(S)] for _ in range(S)]

    def add_BC(self, component, axis, x, v):
        self.BC_mat = self.BC_mat if self.BC_mat is not None else self.get_empty_operator_matrix()

        base = self.axes[axis]
        assert x in [-1, 1]
        assert type(base) == ChebychovHelper

        O = self.get_Id() * 0
        O[-1, :] = base.get_Dirichlet_BC_row_T(x)

        return O

    def convert_operator_matrix_to_operator(self, M):
        if len(self.components) == 1:
            return M[0]
        else:
            return self.sparse_lib.bmat(M)

    def get_grid(self):
        return self.xp.meshgrid(*[me.get_1dgrid() for me in self.axes[::-1]])

    def setup_fft(self, useMPI=False, comm=None):
        if len(self.axes) > 1:
            assert all(
                type(me) != ChebychovHelper for me in self.axes[:-1]
            ), 'Due to handling of imaginary part, we can only have Chebychov in the last dimension!'

        shape = [me.N for me in self.axes]
        self.fft = {}
        self.ifft = {}

        bases = [ChebychovHelper, FFTHelper]

        indeces = np.array([-i - 1 for i in range(len(self.axes))][::-1])
        axes = [tuple(indeces), tuple(indeces[::-1])]
        for base in bases:
            mask = [i for i in indeces if type(self.axes[i]) == base]
            if len(mask) > 0:
                axes.append(tuple(indeces[mask]))
                if len(indeces[mask]) > 1:
                    axes.append(tuple(indeces[mask][::-1]))

        for axis in axes:

            if useMPI:
                from mpi4py import MPI
                from mpi4py_fft import PFFT

                comm = comm if comm else MPI.COMM_WORLD

                _fft = PFFT(
                    comm=comm,
                    shape=shape,
                    axes=axis,
                    dtype='D',
                    collapse=True,
                    # backend=self.fft_backend,
                    # comm_backend=self.fft_comm_backend,
                )
                self.fft[axis] = _fft.forward
                self.ifft[axis] = _fft.backward
            else:
                self.fft[axis] = self.xp.fft.fftn
                self.ifft[axis] = self.xp.fft.ifftn

    def _transform_fft(self, u, axes):
        return self.fft[axes](u, axes=axes)

    def _transform_dct(self, u, axes):
        result = u.copy()

        v = u.copy()

        for axis in axes:
            slices = [slice(0, s, 1) for s in u.shape]
            slices[axis] = self.axes[axis].fft_utils['fwd']['shuffle']
            v = v[*slices]

        V = self.fft[axes](v, axes=axes)

        for axis in axes:
            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)
            V *= self.axes[axis].fft_utils['fwd']['shift'][*expansion]

        result.real[...] = V.real[...]
        return result

    def transform(self, u, axes):
        trfs = {
            ChebychovHelper: self._transform_dct,
            FFTHelper: self._transform_fft,
        }

        result = u.copy()

        axes_base = []
        for base in trfs.keys():
            axes_base = tuple(me for me in axes if type(self.axes[me]) == base)
            if len(axes_base) > 0:
                result = trfs[base](result, axes=axes_base)

        return result

    def _transform_ifft(self, u, axes):
        return self.ifft[axes](u, axes=axes)

    def _transform_idct(self, u, axes):
        result = u.copy()

        v = self.xp.empty(u.shape, dtype='complex')
        v[...] = u[...]

        for axis in axes:
            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            v *= self.axes[axis].fft_utils['bck']['shift'][*expansion]

        V = self.ifft[axes](v, axes=axes)

        for axis in axes:
            slices = [slice(0, s, 1) for s in u.shape]
            slices[axis] = self.axes[axis].fft_utils['bck']['shuffle']
            V = V[*slices]

        result.real[...] = V.real[...]
        return result

    def itransform(self, u, axes):
        trfs = {
            FFTHelper: self._transform_ifft,
            ChebychovHelper: self._transform_idct,
        }

        result = u.copy()

        for base in trfs.keys():
            axes_base = tuple(me for me in axes if type(self.axes[me]) == base)
            if len(axes_base) > 0:
                result = trfs[base](result, axes=axes_base)

        return result

    def get_differentiation_matrix(self, axes):
        """
        Get differentiation matrix along specified axis.

        Args:
            axes (tuple): Axes along which to differentiate.

        Returns:
            sparse differentiation matrix
        """
        sp = self.sparse_lib
        D = sp.eye(np.prod([me.N for me in self.axes]), dtype=complex).tolil() * 0
        ndim = len(self.axes)

        if ndim == 1:
            D = self.axes[0].get_differentiation_matrix()
        elif ndim == 2:
            for axis in axes:
                axis2 = (axis + 1) % ndim
                D1D = self.axes[axis].get_differentiation_matrix()

                if len(axes) > 1:
                    I1D = sp.eye(self.axes[axis2].N)
                else:
                    I1D = self.axes[axis2].get_Id()

                mats = [None] * ndim
                mats[axis] = D1D
                mats[axis2] = I1D

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
        S = sp.eye(np.prod([me.N for me in self.axes]), dtype=complex).tolil() * 0
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
                mats[axis] = S1D
                mats[axis2] = I1D

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
        ndim = len(self.axes)

        if ndim == 1:
            I = self.axes[0].get_Id()
        elif ndim == 2:
            I = sp.kron(*[me.get_Id() for me in self.axes])
        else:
            raise NotImplementedError(f'Basis change matrix not implemented for {ndim} dimension!')

        return I

    def get_basis_change_matrix(self):
        """
        Some spectral bases do a change between bases while differentiating. You can use this matrix to change back to
        the original basis afterwards.

        Returns:
            sparse basis change matrix
        """
        sp = self.sparse_lib
        C = sp.eye(np.prod([me.N for me in self.axes]), dtype=complex).tolil()
        ndim = len(self.axes)

        if ndim == 1:
            C = self.axes[0].get_basis_change_matrix()
        elif ndim == 2:
            mats = [None] * ndim
            mats[0] = self.axes[0].get_basis_change_matrix()
            mats[1] = self.axes[1].get_basis_change_matrix()

            C = C @ sp.kron(*mats)
        else:
            raise NotImplementedError(f'Basis change matrix not implemented for {ndim} dimension!')

        return C
