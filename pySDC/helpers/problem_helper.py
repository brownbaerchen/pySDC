import numpy as np
import scipy
from scipy.special import factorial


def get_steps(derivative, order, stencil_type):
    """
    Get the offsets for the FD stencil.

    Args:
        derivative (int): Order of the derivative
        order (int): Order of accuracy
        stencil_type (str): Type of the stencil
        steps (list): Provide specific steps, overrides `stencil_type`

    Returns:
        int: The number of elements in the stencil
        numpy.ndarray: The offsets for the stencil
    """
    if stencil_type == 'center':
        n = order + derivative - (derivative + 1) % 2 // 1
        steps = np.arange(n) - n // 2
    elif stencil_type == 'forward':
        n = order + derivative
        steps = np.arange(n)
    elif stencil_type == 'backward':
        n = order + derivative
        steps = -np.arange(n)
    elif stencil_type == 'upwind':
        n = order + derivative

        if n <= 3:
            n, steps = get_steps(derivative, order, 'backward')
        else:
            steps = np.append(-np.arange(n - 1)[::-1], [1])
    else:
        raise ValueError(
            f'Stencil must be of type "center", "forward", "backward" or "upwind", not {stencil_type}. If you want something else you can also give specific steps.'
        )
    return n, steps


def get_finite_difference_stencil(derivative, order=None, stencil_type=None, steps=None):
    """
    Derive general finite difference stencils from Taylor expansions

    Args:
        derivative (int): Order of the derivative
        order (int): Order of accuracy
        stencil_type (str): Type of the stencil
        steps (list): Provide specific steps, overrides `stencil_type`

    Returns:
        numpy.ndarray: The weights of the stencil
        numpy.ndarray: The offsets for the stencil
    """

    if steps is not None:
        n = len(steps)
    else:
        n, steps = get_steps(derivative, order, stencil_type)

    # make a matrix that contains the Taylor coefficients
    A = np.zeros((n, n))
    idx = np.arange(n)
    inv_facs = 1.0 / factorial(idx)
    for i in range(0, n):
        A[i, :] = steps ** idx[i] * inv_facs[i]

    # make a right hand side vector that is zero everywhere except at the position of the desired derivative
    sol = np.zeros(n)
    sol[derivative] = 1.0

    # solve the linear system for the finite difference coefficients
    coeff = np.linalg.solve(A, sol)

    # sort coefficients and steps
    coeff = coeff[np.argsort(steps)]
    steps = np.sort(steps)

    return coeff, steps


def get_finite_difference_matrix(
    derivative,
    order,
    stencil_type=None,
    steps=None,
    dx=None,
    size=None,
    dim=None,
    bc=None,
    cupy=False,
    bc_params=None,
):
    """
    Build FD matrix from stencils, with boundary conditions.
    Keep in mind that the boundary conditions may require further modification of the right hand side.

    Args:
        derivative (int): Order of the spatial derivative
        order (int): Order of accuracy
        stencil_type (str): Type of stencil
        steps (list): Provide specific steps, overrides `stencil_type`
        dx (float): Mesh width
        size (int): Number of degrees of freedom per dimension
        dim (int): Number of dimensions
        bc (str): Boundary conditions for both sides
        cupy (bool): Construct a GPU ready matrix if yes

    Returns:
        Sparse matrix: Finite difference matrix
        numpy.ndarray: Vector containing information about the boundary conditions
    """
    if cupy:
        import cupyx.scipy.sparse as sp
    else:
        import scipy.sparse as sp

    # get stencil
    coeff, steps = get_finite_difference_stencil(
        derivative=derivative, order=order, stencil_type=stencil_type, steps=steps
    )

    if type(bc) is not tuple:
        assert type(bc) == str, 'Please pass BCs as string or tuple of strings'
        bc = (bc, bc)
    bc_params = bc_params if bc_params is not None else {}
    if type(bc_params) is not list:
        bc_params = [bc_params, bc_params]

    b = np.zeros(size**dim)

    if bc[0] == 'periodic':
        assert bc[1] == 'periodic'
        A_1d = 0 * sp.eye(size, format='csc')
        for i in steps:
            A_1d += coeff[i] * sp.eye(size, k=steps[i])
            if steps[i] > 0:
                A_1d += coeff[i] * sp.eye(size, k=-size + steps[i])
            if steps[i] < 0:
                A_1d += coeff[i] * sp.eye(size, k=size + steps[i])
    else:
        A_1d = sp.diags(coeff, steps, shape=(size, size), format='lil')

        # Default parameters for Dirichlet and Neumann BCs
        bc_params_defaults = {
            'val': 0.0,
            'neumann_bc_order': order,
            'reduce': False,
        }

        # Loop over each side (0 for left, 1 for right)
        for iS in [0, 1]:
            # -- check Boundary condition types
            assert "neumann" in bc[iS] or "dirichlet" in bc[iS], f"unknown BC type : {bc[iS]}"

            # -- boundary condition parameters
            bc_params[iS] = {**bc_params_defaults, **bc_params[iS]}
            par = bc_params[iS].copy()

            # -- extract parameters and raise an error if additionals
            val = par.pop('val')
            reduce = par.pop('reduce')
            neumann_bc_order = par.pop('neumann_bc_order')
            assert len(par) == 0, f"unused BCs parameters : {par}"

            # -- half stencil width
            sWidth = -min(steps) if iS == 0 else max(steps)

            # -- loop over lines of A that have to be modified
            for i in range(sWidth):
                # -- index of the line
                iLine = i if iS == 0 else -i - 1
                # -- slice for coefficients used in the A matrix
                sCoeff = slice(1, None) if iS == 0 else slice(None, -1)
                # -- index of coefficient used in the b vector
                iCoeff = 0 if iS == 0 else -1

                if reduce:
                    # -- reduce order close to boundary
                    b_coeff, b_steps = get_finite_difference_stencil(
                        derivative=derivative,
                        order=2 * (i + 1),
                        stencil_type='center',
                    )
                else:
                    # -- shift stencil close to boundary
                    b_steps = (
                        np.arange(-(i + 1), order + derivative - (i + 1))
                        if iS == 0
                        else np.arange(-(order + derivative) + (i + 2), (i + 2))
                    )

                    b_coeff, b_steps = get_finite_difference_stencil(derivative=derivative, steps=b_steps)

                # -- column slice where to put coefficients in the A matrix
                colSlice = slice(None, len(b_coeff) - 1) if iS == 0 else slice(-len(b_coeff) + 1, None)

                # -- modify A
                A_1d[iLine, :] = 0
                A_1d[iLine, colSlice] = b_coeff[sCoeff]

                if "dirichlet" in bc[iS]:
                    # -- modify b
                    b[iLine] = val * b_coeff[iCoeff]

                elif "neumann" in bc[iS]:
                    nOrder = neumann_bc_order

                    # -- generate the first derivative stencil
                    n_coeff, n_steps = get_finite_difference_stencil(
                        derivative=1, order=nOrder, stencil_type="forward" if iS == 0 else "backward"
                    )

                    # -- column slice where to put coefficients in the A matrix
                    colSlice = slice(None, len(n_coeff) - 1) if iS == 0 else slice(-len(n_coeff) + 1, None)

                    # -- additional modification to A
                    A_1d[iLine, colSlice] -= b_coeff[iCoeff] / n_coeff[iCoeff] * n_coeff[sCoeff]

                    # -- modify B
                    b[iLine] = val * b_coeff[iCoeff] / n_coeff[iCoeff] * dx

    # TODO: extend the BCs to higher dimensions
    A_1d = A_1d.tocsc()
    if dim == 1:
        A = A_1d
    elif dim == 2:
        A = sp.kron(A_1d, sp.eye(size)) + sp.kron(sp.eye(size), A_1d)
    elif dim == 3:
        A = (
            sp.kron(A_1d, sp.eye(size**2))
            + sp.kron(sp.eye(size**2), A_1d)
            + sp.kron(sp.kron(sp.eye(size), A_1d), sp.eye(size))
        )
    else:
        raise NotImplementedError(f'Dimension {dim} not implemented.')

    A /= dx**derivative
    b /= dx**derivative

    return A, b


def get_1d_grid(size, bc, left_boundary=0.0, right_boundary=1.0):
    """
    Generate a grid in one dimension and obtain mesh spacing for finite difference discretization.

    Args:
        size (int): Number of degrees of freedom per dimension
        bc (str): Boundary conditions for both sides
        left_boundary (float): x value at the left boundary
        right_boundary (float): x value at the right boundary

    Returns:
        float: mesh spacing
        numpy.ndarray: 1d mesh
    """
    L = right_boundary - left_boundary
    if bc == 'periodic':
        dx = L / size
        xvalues = np.array([left_boundary + dx * i for i in range(size)])
    elif "dirichlet" in bc or "neumann" in bc:
        dx = L / (size + 1)
        xvalues = np.array([left_boundary + dx * (i + 1) for i in range(size)])
    else:
        raise NotImplementedError(f'Boundary conditions \"{bc}\" not implemented.')

    return dx, xvalues


class SpectralHelper:
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


class ChebychovHelper(SpectralHelper):

    def __init__(self, *args, S=1, d=1, mode='T2U', **kwargs):
        super().__init__(*args, **kwargs)
        self.S = S
        self.d = d
        self.mode = mode

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
            return self.get_U2T_integration_matrix()
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

    def get_T2U_differentiation_matrix(self):
        '''
        Sparse differentiation matrix from Chebychov polynomials of first kind to second. When using this, you must
        formulate your problem in first order derivatives.

        Returns:
            scipy.sparse: Sparse differentiation matrix
        '''
        return self.sparse_lib.diags(self.xp.arange(self.N - 1) + 1, offsets=1, format=self.sparse_format)

    def get_U2T_integration_matrix(self):
        S = self.sparse_lib.diags(1 / (self.xp.arange(self.N - 1) + 1), offsets=-1, format=self.sparse_format).tolil()
        S[0, 1::2] = 1 / (2 ** self.xp.arange(self.N))[1::2]
        return S

    def get_T2T_integration_matrix(self):
        # S = self.sparse_lib.diags(0.5 / (self.xp.arange(self.N - 1) + 1), offsets=-1, format=self.sparse_format) - self.sparse_lib.diags(0.5 / (self.xp.arange(self.N - 1) - 1), offsets=1, format=self.sparse_format)
        return self.get_U2T_integration_matrix() @ self.get_conv('T2U')

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

    def transform(self, u, axis=-1):
        return self.fft_lib.dct(u, axis=axis) * self.norm

    def itransform(self, u, axis=-1):
        assert self.norm.shape[0] == u.shape[axis]

        return self.fft_lib.idct(u / self.norm, axis=axis)
        return self.fft_lib.idct(u / self.norm.reshape(u.shape), axis=axis)

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


class FFTHelper(SpectralHelper):

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

    def transform(self, u, axis=-1):
        return self.fft_lib.fft(u, axis=axis)

    def itransform(self, u, axis=-1):
        return self.fft_lib.ifft(u, axis=axis)
