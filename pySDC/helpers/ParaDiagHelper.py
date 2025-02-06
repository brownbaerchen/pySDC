import numpy as np


def get_fft_matrix(N):
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
