import numpy as np
import scipy
from pySDC.implementations.datatype_classes.mesh import mesh


class SpectralHelper1D:
    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    xp = np

    def __init__(self, N, x0=None, x1=None, **kwargs):
        self.N = N
        self.x0 = x0
        self.x1 = x1
        self.L = x1 - x0

    def get_Id(self):
        raise NotImplementedError

    def get_zero(self):
        return 0 * self.get_Id()

    def get_differentiation_matrix(self):
        raise NotImplementedError()

    def get_integration_matrix(self):
        raise NotImplementedError()

    def get_wavenumbers(self):
        raise NotImplementedError

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

    def get_BC(self, kind, **kwargs):
        raise NotImplementedError(f'No boundary conditions of {kind=!r} implemented!')

    def get_filter_matrix(self, kmin=0, kmax=None):
        k = abs(self.get_wavenumbers())

        kmax = max(k) if kmax is None else kmax

        mask = self.xp.logical_or(k >= kmax, k < kmin)

        F = self.get_Id().tolil()
        F[:, mask] = 0
        return F.tocsc()

    def get_1dgrid(self):
        raise NotImplementedError


class ChebychovHelper(SpectralHelper1D):
    def __init__(self, *args, S=1, d=1, mode='T2U', transform_type='fft', x0=-1, x1=1, **kwargs):
        assert x0 == -1
        assert x1 == 1
        super().__init__(*args, x0=x0, x1=x1, **kwargs)
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

    def get_integration_matrix(self, lbnd=0):
        if self.mode == 'T2T':
            return self.get_T2T_integration_matrix(lbnd=lbnd)
        elif self.mode == 'T2U':
            return self.get_conv('T2U') @ self.get_T2T_integration_matrix(lbnd=lbnd)
        else:
            raise NotImplementedError(f'{self.mode=!r} not implemented')

    def get_wavenumbers(self):
        return self.xp.arange(self.N)

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
        elif direction == 'backward':
            return self.get_conv(self.mode[::-1])
        else:
            return self.get_conv(direction)

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

    def get_T2T_integration_matrix(self, lbnd=0):
        # TODO: this is a bit fishy
        S = (self.get_U2T_integration_matrix() @ self.get_conv('T2U')).tolil()
        n = self.xp.arange(self.N)
        if lbnd == 0:
            S[0, 1::2] = (
                (n / (2 * (self.xp.arange(self.N) + 1)))[1::2]
                * (-1) ** (self.xp.arange(self.N // 2))
                / (np.append([1], self.xp.arange(self.N // 2 - 1) + 1))
            )
        else:
            raise NotImplementedError
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

    def get_norm(self, N=None):
        '''get normalization for converting Chebychev coefficients and DCT'''
        N = self.N if N is None else N
        norm = self.xp.ones(N) / N
        norm[0] /= 2
        return norm

    def get_fft_shuffle(self, forward, N):
        xp = self.xp
        if forward:
            return xp.append(xp.arange((N + 1) // 2) * 2, -xp.arange(N // 2) * 2 - 1 - N % 2)
        else:
            mask = xp.zeros(N, dtype=int)
            mask[: N - N % 2 : 2] = xp.arange(N // 2)
            mask[1::2] = N - xp.arange(N // 2) - 1
            mask[-1] = N // 2
            return mask

    def get_fft_shift(self, forward, N):
        k = self.get_wavenumbers()
        norm = self.get_norm()
        xp = self.xp
        if forward:
            return 2 * xp.exp(-1j * np.pi * k / (2 * N) + 0j * np.pi / 4) * norm
        else:
            shift = xp.exp(1j * np.pi * k / (2 * N))
            shift[0] /= 2
            return shift / norm

    def get_fft_utils(self):
        self.fft_utils = {
            'fwd': {},
            'bck': {},
        }
        xp = self.xp

        # forwards transform
        self.fft_utils['fwd']['shuffle'] = self.get_fft_shuffle(True, self.N)
        self.fft_utils['fwd']['shift'] = self.get_fft_shift(True, self.N)

        # backwards transform
        self.fft_utils['bck']['shuffle'] = self.get_fft_shuffle(False, self.N)
        self.fft_utils['bck']['shift'] = self.get_fft_shift(False, self.N)

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

    def get_BC(self, kind, **kwargs):
        if kind.lower() == 'integral':
            return self.get_integ_BC_row_T(**kwargs)
        elif kind.lower() == 'dirichlet':
            return self.get_Dirichlet_BC_row_T(**kwargs)
        else:
            return super().get_BC(kind, **kwargs)

    def get_integ_BC_row_T(self, **kwargs):
        """
        Get a row for generating integral BCs with T polynomials.
        It returns the values of the T polynomials at x.

        Returns:
            self.xp.ndarray: Row to put into a matrix
        """
        n = self.xp.arange(self.N) + 1
        me = self.xp.zeros_like(n).astype(float)
        me[2:] = ((-1) ** n[1:-1] + 1) / (1 - n[1:-1] ** 2)
        me[0] = 2.0
        return me

    def get_Dirichlet_BC_row_T(self, x, **kwargs):
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
    def __init__(self, *args, x0=0, x1=2 * np.pi, **kwargs):
        super().__init__(*args, x0=x0, x1=x1, **kwargs)

    def get_1dgrid(self):
        dx = self.L / self.N
        return self.xp.arange(self.N) * dx + self.x0

    def get_wavenumbers(self):
        return self.xp.fft.fftfreq(self.N, 1.0 / self.N) * 2 * np.pi / self.L

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

    def get_BC(self, kind, **kwargs):
        if kind.lower() == 'integral':
            return self.get_integ_BC_row(**kwargs)
        else:
            return super().get_BC(kind, **kwargs)

    def get_integ_BC_row(self, **kwargs):
        me = self.xp.zeros(self.N)
        me[0] = self.L / self.N
        return me


class SpectralHelper:
    xp = np
    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    dtype = mesh
    fft_backend = 'fftw'
    fft_comm_backend = 'MPI'

    @classmethod
    def setup_GPU(cls):
        """switch to GPU modules"""
        import cupy as cp
        import cupyx.scipy.sparse as sparse_lib

        cls.xp = cp
        cls.sparse_lib = sparse_lib

        cls.fft_backend = 'cupy'
        cls.fft_comm_backend = 'NCCL'

    def __init__(self, comm=None, useGPU=False, debug=False):
        self.comm = comm
        self.debug = debug
        if useGPU:
            self.setup_GPU()

        self.axes = []
        self.components = []

        self.full_BCs = []
        self.BC_mat = None
        self.BCs = None
        self.tau_terms_in_equation = {}
        # self.boundary_bordering=boundary_bordering

        self.fft_cache = {}

    @property
    def u_init(self):
        return self.dtype(self.init)

    @property
    def u_init_forward(self):
        return self.dtype(self.init_forward)

    @property
    def shape(self):
        return self.init[0][1:]

    @property
    def ndim(self):
        return len(self.axes)

    @property
    def ncomponents(self):
        return len(self.components)

    def add_axis(self, base, *args, **kwargs):
        if base.lower() in ['chebychov', 'chebychev', 'cheby', 'chebychovhelper']:
            kwargs['transform_type'] = kwargs.get('transform_type', 'fft')
            self.axes.append(ChebychovHelper(*args, **kwargs))
        elif base.lower() in ['fft', 'fourier', 'ffthelper']:
            self.axes.append(FFTHelper(*args, **kwargs))
        else:
            raise NotImplementedError(f'{base=!r} is not implemented!')
        self.axes[-1].xp = self.xp
        self.axes[-1].sparse_lib = self.sparse_lib

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

    def get_BC(self, axis, kind, line=-1, pressure_gauge_FFT=False, **kwargs):
        base = self.axes[axis]

        BC = base.get_Id().tolil() * 0
        BC[line, :] = base.get_BC(kind=kind, **kwargs)

        ndim = len(self.axes)
        if ndim == 1:
            return BC
        elif ndim == 2:
            axis2 = (axis + 1) % ndim

            if pressure_gauge_FFT:
                _Id = self.sparse_lib.diags(self.xp.append([1], self.xp.zeros(self.axes[axis2].N - 1)))
            else:
                _Id = self.axes[axis2].get_Id().tolil()

            Id = self.get_local_slice_of_1D_matrix(self.axes[axis2].get_basis_change_matrix() @ _Id, axis=axis2)
            mats = [
                None,
            ] * ndim
            mats[axis] = self.get_local_slice_of_1D_matrix(BC, axis=axis)
            mats[axis2] = Id
            return self.sparse_lib.kron(*mats)

    #     def add_gauge(self, component, equation):
    #         _BC = self.get_BC(axis=axis, kind=kind, line=line, **kwargs) * scale
    #

    def add_BC(
        self, component, equation, axis, kind, v, zero_line=True, scale=1.0, line=-1, pressure_gauge_FFT=False, **kwargs
    ):
        self.tau_terms_in_equation[equation] = self.tau_terms_in_equation.get(equation, 0) + 1
        line = line  # -self.tau_terms_in_equation[equation]

        _BC = self.get_BC(axis=axis, kind=kind, line=line, pressure_gauge_FFT=pressure_gauge_FFT, **kwargs) * scale
        self.BC_mat[self.index(equation)][self.index(component)] += _BC
        self.full_BCs += [
            {
                'component': component,
                'equation': equation,
                'axis': axis,
                'kind': kind,
                'v': v * scale,
                'zero_line': zero_line,
                'line': line,
                **kwargs,
            }
        ]

        if zero_line:
            if pressure_gauge_FFT:
                slices = [self.index(equation)] + [
                    line,
                ] * self.ndim
            else:
                slices = (
                    [self.index(equation)]
                    + [slice(0, self.init[0][i + 1]) for i in range(axis)]
                    + [line]
                    + [slice(0, self.init[0][i + 1]) for i in range(axis + 1, len(self.axes))]
                )
            N = self.axes[axis].N
            if (N + line) % N in self.xp.arange(N)[self.local_slice[axis]]:
                self.BC_rhs_mask[*slices] = True

    def setup_BCs(self):
        BC = self.convert_operator_matrix_to_operator(self.BC_mat)

        self.BC_mask = BC != 0
        self._BCs = BC.tolil()[self.BC_mask]
        self.BC_zero_index = self.xp.arange(np.prod(self.init[0]))[self.BC_rhs_mask.flatten()]

    def check_BCs(self, u):
        for axis in range(self.ndim):
            BCs = [me for me in self.full_BCs if me["axis"] == axis]

            if len(BCs) > 0:
                u_hat = self.transform(u, axes=(axis - self.ndim,))
                for BC in BCs:
                    get = u_hat[self.index(BC['component'])] @ self.axes[axis].get_BC(**BC)
                    want = BC['v']
                    assert self.xp.allclose(
                        get, want
                    ), f'Unexpected BC in {BC["component"]} in equation {BC["equation"]}! Got {get}, wanted {want}'

    def put_BCs_in_matrix(self, A, rescale=1.0):
        A = A.tolil()
        A[self.BC_zero_index, :] = 0  # TODO: Smells like tuna
        A[self.BC_mask] = self._BCs * rescale
        # if not self.boundary_bordering:
        #     A[:, self.BC_zero_index] = 0
        #     for BC in self.full_BCs:
        #         idx = self.components.index(BC['equation'])
        #         A[idx * np.prod(self.shape), (idx+1) * np.prod(self.shape)-1] = 1
        #         # A[(idx+1) * np.prod(self.shape)-2, (idx+1) * np.prod(self.shape)-1] = 1
        return A.tocsc()

    # def padd_with_tau_terms(self, u_hat):
    #     padded_shape = list(u_hat.shape)
    #     for axis in range(self.ndim):
    #         padded_shape[axis+1] += max([0] - [BC['line'] for BC in self.full_BCs if BC['axis'] == axis])
    #     me = self.xp.zeros(shape=padded_shape, dtype=complex)

    #     slices = [slice(0, me) for me in u_hat.shape]
    #     print(slices, u_hat.shape)
    #     me[*slices] = u_hat
    #     return me

    # def remove_tau_terms(self, u_hat):
    #     if self.boundary_bordering:
    #         return u_hat
    #     else:
    #         for BC in self.full_BCs:
    #             slices = [slice(0, me.N) for me in self.axes]
    #             slices[BC['axis']] = BC['line']
    #             u_hat[*slices] = 0
    #         return u_hat

    def put_BCs_in_rhs(self, rhs, istransformed=False, rescale=1.0):
        assert rhs.ndim > 1, 'rhs must not be flattened here!'

        ndim = len(self.axes)

        for axis in range(ndim):
            if istransformed:
                _rhs_hat = rhs
            else:
                _rhs_hat = self.transform(rhs, axes=(axis - ndim,))

            for bc in self.full_BCs:

                slices = (
                    [slice(0, self.init[0][i + 1]) for i in range(axis)]
                    + [bc['line']]
                    + [slice(0, self.init[0][i + 1]) for i in range(axis + 1, len(self.axes))]
                )
                if axis == bc['axis']:
                    _slice = [self.index(bc['equation'])] + slices
                    if bc['zero_line']:
                        _rhs_hat[*_slice] = bc['v'] * rescale
                    else:
                        _rhs_hat[*_slice] += bc['v'] * rescale

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

    def get_wavenumbers(self):
        grids = [self.axes[i].get_wavenumbers()[self.local_slice[i]] for i in range(len(self.axes))][::-1]
        return self.xp.meshgrid(*grids)

    def get_grid(self):
        grids = [self.axes[i].get_1dgrid()[self.local_slice[i]] for i in range(len(self.axes))][::-1]
        return self.xp.meshgrid(*grids)

    def get_fft(self, axes=None, direction='object', padding=None, shape=None, comm=None):
        axes = tuple(-i - 1 for i in range(self.ndim)) if axes is None else axes
        shape = self.global_shape[1:] if shape is None else shape
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )
        key = (axes, direction, tuple(padding), tuple(shape))

        if key not in self.fft_cache.keys():
            if self.comm is None:
                assert np.allclose(padding, 1), 'Zero padding is not implemented for non-MPI transforms'

                if direction == 'forward':
                    self.fft_cache[key] = self.xp.fft.fftn
                elif direction == 'backward':
                    self.fft_cache[key] = self.xp.fft.ifftn
                elif direction == 'object':
                    self.fft_cache[key] = None
            else:
                from mpi4py_fft import PFFT, newDistArray

                _fft = PFFT(
                    comm=self.comm if comm is None else comm,
                    shape=shape,
                    axes=sorted(list(axes)),
                    dtype='D',
                    collapse=False,
                    backend=self.fft_backend,
                    comm_backend=self.fft_comm_backend,
                    padding=padding,
                )
                if direction == 'forward':
                    self.fft_cache[key] = _fft.forward
                elif direction == 'backward':
                    self.fft_cache[key] = _fft.backward
                elif direction == 'object':
                    self.fft_cache[key] = _fft

        return self.fft_cache[key]

    def setup_fft(self):

        if len(self.components) == 0:
            self.add_component('u')

        shape = [me.N for me in self.axes]
        self.global_shape = (len(self.components),) + tuple(me.N for me in self.axes)
        self.local_slice = [slice(0, me.N) for me in self.axes]

        axes = tuple(i for i in range(len(self.axes)))
        self.fft_obj = self.get_fft(axes=axes, direction='object')
        if self.fft_obj is not None:
            self.local_slice = self.fft_obj.local_slice(False)

        self.init = (np.empty(shape=self.global_shape)[:, *self.local_slice].shape, self.comm, np.dtype('float'))
        self.init_forward = (
            np.empty(shape=self.global_shape)[:, *self.local_slice].shape,
            self.comm,
            np.dtype('complex128'),
        )

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

    def _transform_fft(self, u, axes, **kwargs):
        fft = self.get_fft(axes, 'forward', **kwargs)
        return fft(u, axes=axes)

    def _transform_dct(self, u, axes, padding=None, **kwargs):
        '''
        This will only ever return real values!
        '''
        if self.debug:
            assert self.xp.allclose(u.imag, 0), 'This function can only handle real input.'

        if len(axes) > 1:
            v = self._transform_dct(self._transform_dct(u, axes[1:], **kwargs), (axes[0],), **kwargs)
        else:
            v = u.copy().astype(complex)
            axis = axes[0]
            base = self.axes[axis]

            shuffle = [slice(0, s, 1) for s in u.shape]
            shuffle[axis] = base.get_fft_shuffle(True, N=v.shape[axis])
            v = v[*shuffle]

            if padding is not None:
                shape = list(v.shape)
                if self.comm:
                    shape[0] = self.comm.allreduce(v.shape[0])
                fft = self.get_fft(axes, 'forward', shape=shape)
            else:
                fft = self.get_fft(axes, 'forward', **kwargs)

            v = fft(v, axes=axes)

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, v.shape[axis], 1)

            if padding is not None:
                shift = base.get_fft_shift(True, v.shape[axis])

                if padding[axis] != 1:
                    N = int(np.ceil(v.shape[axis] / padding[axis]))
                    _expansion = [slice(0, n) for n in v.shape]
                    _expansion[axis] = slice(0, N, 1)
                    v = v[*_expansion]
            else:
                shift = base.fft_utils['fwd']['shift']

            v *= shift[*expansion]

        return v.real

    def transform_single_component(self, u, axes=None, padding=None):
        trfs = {
            ChebychovHelper: self._transform_dct,
            FFTHelper: self._transform_fft,
        }

        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )  # You know, sometimes I feel very strongly about Black still. This atrocious formatting is readable by Sauron only.

        result = u.copy().astype(complex)
        alignment = self.ndim - 1

        axes_collapsed = [tuple(sorted(me for me in axes if type(self.axes[me]) == base)) for base in trfs.keys()]
        bases = [list(trfs.keys())[i] for i in range(len(axes_collapsed)) if len(axes_collapsed[i]) > 0]
        axes_collapsed = [me for me in axes_collapsed if len(me) > 0]
        shape = [max(u.shape[i], self.global_shape[1 + i]) for i in range(self.ndim)]

        fft = self.get_fft(axes=axes, padding=padding, direction='object')
        if fft is not None:
            shape = list(fft.global_shape(False))

        for trf in range(len(axes_collapsed)):
            _axes = axes_collapsed[trf]
            base = bases[trf]

            if len(_axes) == 0:
                continue

            for _ax in _axes:
                shape[_ax] = self.global_shape[1 + self.ndim + _ax]

            fft = self.get_fft(_axes, 'object', padding=padding, shape=shape)

            _in = self.get_aligned(result, axis_in=alignment, axis_out=self.ndim + _axes[-1], forward=False, fft=fft)

            alignment = self.ndim + _axes[-1]

            _out = trfs[base](_in, axes=_axes, padding=padding, shape=shape)

            if self.comm is not None:
                _out *= np.prod([self.axes[i].N for i in _axes])

            axes_next_base = axes_collapsed[(trf + 1) % len(axes_collapsed)]
            alignment = alignment if len(axes_next_base) == 0 else self.ndim + axes_next_base[-1]
            result = self.get_aligned(_out, axis_in=self.ndim + _axes[0], axis_out=alignment, fft=fft, forward=True)

        fft = self.get_fft(axes=axes, padding=padding)
        return self.get_aligned(result, axis_in=alignment, axis_out=self.ndim - 1, fft=fft, forward=True)

    def transform(self, u, axes=None, padding=None):
        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )

        alignment = self.ndim - 1
        result = [
            None,
        ] * self.ncomponents
        for comp in self.components:
            i = self.index(comp)

            result[i] = self.transform_single_component(u[i], axes=axes, padding=padding)

        return self.xp.array(result)

    # def transform(self, u, axes=None, padding=None):
    #     trfs = {
    #         ChebychovHelper: self._transform_dct,
    #         FFTHelper: self._transform_fft,
    #     }

    #     axes = tuple(-i - 1 for i in range(self.ndim)) if axes is None else axes

    #     result = u.copy().astype(complex)
    #     alignment = self.ndim - 1

    #     for comp in self.components:
    #         i = self.index(comp)

    #         for base in trfs.keys():
    #             axes_base = tuple(sorted(me for me in axes if type(self.axes[me]) == base))

    #             if len(axes_base) > 0:
    #                 _in = self.get_aligned(result[i], axis_in=alignment, axis_out=self.ndim + axes_base[-1])

    #                 fft = self.get_fft(axes_base, 'object', padding=padding)
    #                 if fft:
    #                     from mpi4py_fft import newDistArray

    #                     _out = newDistArray(fft, True)
    #                 else:
    #                     _out = _in

    #                 _out[...] = trfs[base](_in, axes=axes_base)

    #                 if fft:
    #                     result[i] = _out.redistribute(-1) * np.prod([self.axes[i].N for i in axes_base])
    #                 else:
    #                     result[i] = _out

    #     return result

    def _transform_ifft(self, u, axes, **kwargs):
        ifft = self.get_fft(axes, 'backward', **kwargs)
        return ifft(u, axes=axes)

    def _transform_idct(self, u, axes, padding=None, **kwargs):
        '''
        This will only ever return real values!
        '''
        if self.debug:
            assert self.xp.allclose(u.imag, 0), 'This function can only handle real input.'

        v = u.copy().astype(complex)

        if len(axes) > 1:
            v = self._transform_idct(self._transform_idct(u, axes[1:]), (axes[0],))
        else:
            axis = axes[0]
            base = self.axes[axis]

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)

            if padding is not None:
                N = int(np.ceil(v.shape[axis] * padding[axis]))
                shift = base.get_fft_shift(False, N)
            else:
                shift = base.fft_utils['bck']['shift']

            v *= shift[*expansion]

            ifft = self.get_fft(axes, 'backward', padding=padding, **kwargs)
            v = ifft(v, axes=axes)

            shuffle = [slice(0, s, 1) for s in v.shape]
            shuffle[axis] = base.get_fft_shuffle(False, N=v.shape[axis])
            v = v[*shuffle]

        return v.real

    def itransform_single_component(self, u, axes=None, padding=None):
        trfs = {
            FFTHelper: self._transform_ifft,
            ChebychovHelper: self._transform_idct,
        }

        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )

        result = u.copy().astype(complex)
        alignment = self.ndim - 1

        axes_collapsed = [tuple(sorted(me for me in axes if type(self.axes[me]) == base)) for base in trfs.keys()]
        bases = [list(trfs.keys())[i] for i in range(len(axes_collapsed)) if len(axes_collapsed[i]) > 0]
        axes_collapsed = [me for me in axes_collapsed if len(me) > 0]
        shape = list(self.global_shape[1:])

        for trf in range(len(axes_collapsed)):
            _axes = axes_collapsed[trf]
            base = bases[trf]

            if len(_axes) == 0:
                continue

            fft = self.get_fft(_axes, 'object', padding=padding, shape=shape)

            _in = self.get_aligned(result, axis_in=alignment, axis_out=self.ndim + _axes[0], forward=True, fft=fft)
            if self.comm is not None:
                _in /= np.prod([self.axes[i].N for i in _axes])

            alignment = self.ndim + _axes[0]

            _out = trfs[base](_in, axes=_axes, padding=padding, shape=shape)

            for _ax in _axes:
                shape[_ax] = _out.shape[_ax]

            axes_next_base = axes_collapsed[(trf + 1) % len(axes_collapsed)]
            alignment = alignment if len(axes_next_base) == 0 else self.ndim + axes_next_base[0]
            result = self.get_aligned(_out, axis_in=self.ndim + _axes[-1], axis_out=alignment, fft=fft, forward=False)

        fft = self.get_fft(axes=axes, padding=padding)
        return self.get_aligned(result, axis_in=alignment, axis_out=self.ndim - 1, fft=fft)

    def get_aligned(self, u, axis_in, axis_out, fft=None, forward=False, fill=True, **kwargs):
        if self.comm is None:
            if fill:
                return u
            elif forward:
                return self.u_init_forward
            else:
                return self.u_init

        from mpi4py_fft import newDistArray

        fft = self.get_fft(**kwargs) if fft is None else fft

        _in = newDistArray(fft, forward).redistribute(axis_in)
        if fill:
            _in[...] = u

        return _in.redistribute(axis_out)

    def itransform(self, u, axes=None, padding=None):
        axes = tuple(-i - 1 for i in range(self.ndim)[::-1]) if axes is None else axes
        padding = (
            [
                1,
            ]
            * self.ndim
            if padding is None
            else padding
        )

        alignment = self.ndim - 1
        result = [
            None,
        ] * self.ncomponents
        for comp in self.components:
            i = self.index(comp)

            result[i] = self.itransform_single_component(u[i], axes=axes, padding=padding)

        return self.xp.array(result)

    def get_local_slice_of_1D_matrix(self, M, axis):
        return M.tolil()[self.local_slice[axis], self.local_slice[axis]]

    # def get_local_slice_of_2D_matrix(self, M, axis):
    #     slices = [slice(0, me) for me in M.shape]
    #     if axis == 0:
    #         slices[axis] = slice(0, M.shape[0], self.local_slice[1])
    #     elif axis == 1:
    #         slices[axis] = self.local_slice[axis]
    #     else:
    #         raise NotImplementedError
    #     print(slices, axis)
    #     return M.tolil()[*slices]

    def get_filter_matrix(self, axis, **kwargs):
        if self.ndim == 1:
            return self.axes[0].get_filter_matrix(**kwargs)

        mats = [base.get_Id() for base in self.axes]
        mats[axis] = self.axes[axis].get_filter_matrix(**kwargs)
        return self.sparse_lib.kron(*mats)

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
            raise NotImplementedError(f'Basis change matrix not implemented for {ndim} dimension!')

        return C

    def get_zero_padded_version(self, axis=0, padding=3 / 2):
        padded = SpectralHelper(self.comm)
        padded.add_component(self.components)
        for i in range(self.ndim):
            base = self.axes[i]
            # assert base.N % 2 == 1
            params = {
                **base.__dict__,
                'N': int(np.ceil(padding * base.N)) if i == axis else base.N,
                'base': type(base).__name__,
            }

            padded.add_axis(**params)
        padded.setup_fft()

        def get_mask(transform, axis):
            K_pad = transform.get_wavenumbers()[::-1]
            k = self.axes[axis].get_wavenumbers()
            mask_top = K_pad[axis] <= self.xp.max(k)
            mask_bottom = K_pad[axis] >= self.xp.min(k)
            return self.xp.logical_and(mask_top, mask_bottom)

        padded.padding_mask_single = get_mask(padded, axis)
        padding_mask = self.xp.zeros(padded.init[0], dtype=bool)
        padding_mask[self.xp.stack([padded.padding_mask_single for _ in range(self.ncomponents)])] = 1
        padded.padding_mask = padding_mask.flatten()
        padded.padding_axis = axis
        padded.unpadded_init = self.init
        return padded

    def get_padded(self, u_hat):
        buffer = self.xp.zeros(np.prod(self.init[0]), dtype=complex)
        buffer[self.padding_mask] = u_hat.flatten()
        return buffer.reshape(self.init[0])

    def get_unpadded(self, u_hat):
        return (u_hat.flatten()[self.padding_mask]).reshape(self.unpadded_init[0])

    def fill_padded(self, u_hat, u_hat_pad):
        assert hasattr(self, 'padding_mask'), 'I don\'t seem to be a padded transform!'

        buffer = self.xp.zeros(shape=np.prod(self.shape), dtype=complex)
        buffer[self.padding_mask] = u_hat.flatten()
        u_hat_pad[...] = buffer.reshape(u_hat_pad.shape)

    def retrieve_padded(self, u_hat, u_hat_pad):
        assert hasattr(self, 'padding_mask'), 'I don\'t seem to be a padded transform!'
        u_hat[...] = (u_hat_pad.flatten()[self.padding_mask]).reshape(u_hat.shape)
