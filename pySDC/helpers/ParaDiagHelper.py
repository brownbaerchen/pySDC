import numpy as np
import scipy.sparse as sp


def get_FFT_matrix(N):
    """
    Get matrix for computing FFT of size N. Normalization is like "ortho" in numpy.
    Compute inverse FFT by multiplying by the complex conjugate (numpy.conjugate) of this matrix

    Args:
        N (int): Size of the data to be transformed

    Returns:
        numpy.ndarray: Dense square matrix to compute forward transform
    """
    idx_1d = np.arange(N, dtype=complex)
    i1, i2 = np.meshgrid(idx_1d, idx_1d)

    return np.exp(-2 * np.pi * 1j * i1 * i2 / N) / np.sqrt(N)


def get_E_matrix(N, alpha=0):
    """
    Get NxN matrix with -1 on the lower subdiagonal, -alpha in the top right and 0 elsewhere

    Args:
        N (int): Size of the matrix
        alpha (float): Negative of value in the top right

    Returns:
        sparse E matrix
    """
    E = sp.diags(
        [
            -1.0,
        ]
        * (N - 1),
        offsets=-1,
    ).tolil()
    E[0, -1] = -alpha
    return E


def get_J_matrix(N, alpha):
    """
    Get matrix for weights in the weighted inverse FFT

    Args:
        N (int): Size of the matrix
        alpha (float): alpha parameter in ParaDiag

    Returns:
        sparse J matrix
    """
    gamma = alpha ** (-np.arange(N) / N)
    return sp.diags(gamma)


def get_J_inv_matrix(N, alpha):
    """
    Get matrix for weights in the weighted FFT

    Args:
        N (int): Size of the matrix
        alpha (float): alpha parameter in ParaDiag

    Returns:
        sparse J_inv matrix
    """
    gamma = alpha ** (-np.arange(N) / N)
    return sp.diags(1 / gamma)


def get_weighted_FFT_matrix(N, alpha):
    """
    Get matrix for the weighted FFT

    Args:
        N (int): Size of the matrix
        alpha (float): alpha parameter in ParaDiag

    Returns:
        Dense weighted FFT matrix
    """
    return get_FFT_matrix(N) @ get_J_inv_matrix(N, alpha)


def get_weighted_iFFT_matrix(N, alpha):
    """
    Get matrix for the weighted inverse FFT

    Args:
        N (int): Size of the matrix
        alpha (float): alpha parameter in ParaDiag

    Returns:
        Dense weighted FFT matrix
    """
    return get_J_matrix(N, alpha) @ np.conjugate(get_FFT_matrix(N))
