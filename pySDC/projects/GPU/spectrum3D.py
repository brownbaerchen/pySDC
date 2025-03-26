import numpy as np
from qmat.lagrange import LagrangeApproximation


def computeMeanSpectrum(uValues, xGrid, zGrid, verbose=False):
    """
    Compute the 3D spectrum of RBC3D fields, interpolating the field on
    the z direction and assuming that it's periodic in that direction.
    Since u(x,y,z=0) = u(x,y,z=L) = 0 because of the non-slip boundary condition,
    it does make sense ...

    Parameters
    ----------
    uValues : np.5darray
        The fields value with shape [nT, 3, nX, nY, nZ], where nT is
        the number of time-steps, 3 corresponds to the 3 velocity components,
        and nX, nY, nZ the 3D shape of the mesh.
    xGrid : np.1darray
        The grid coordinate in x direction (supposedly the same as yGrid).
    zGrid : np.1darray
        The grid coordinate in z direction.
    verbose : bool, optional
        Print computation messages or not. The default is False.

    Returns
    -------
    spectrum : np.1darray
        Spectrum of the velocity field.
    """
    uValues = np.asarray(uValues)
    nT, nVar, *gridSizes = uValues.shape
    assert nVar == len(gridSizes) == 3, "need 3D fields"
    if verbose:
        print(f"Computing Mean Spectrum on u[{', '.join([str(n) for n in uValues.shape])}]")

    # Check for a cube with uniform dimensions
    nX, nY, nZ = gridSizes
    assert nX == nY
    size = nX // 2

    # Interpolate in z direction into a uniform grid
    assert xGrid is not None and zGrid is not None
    if verbose:
        print(" -- interpolating from zGrid to a uniform mesh ...")
    P = LagrangeApproximation(zGrid).getInterpolationMatrix(xGrid)
    uValues = np.einsum('ij,tvxyj->tvxyi', P, uValues)

    # Compute 3D mode shells
    k1D = np.fft.fftfreq(nX, 1 / nX) ** 2
    kMod = k1D[:, None, None] + k1D[None, :, None] + k1D[None, None, :]
    kMod **= 0.5
    idx = kMod.copy()
    idx *= kMod < size
    idx -= kMod >= size
    idxList = range(int(idx.max()) + 1)
    flatIdx = idx.ravel()

    # Fourier transform and square of Im,Re
    if verbose:
        print(" -- 3D FFT on u, v & w ...")
    uHat = np.fft.fftn(uValues, axes=(-3, -2, -1))

    if verbose:
        print(" -- square of Im,Re ...")
    ffts = [uHat[:, i] for i in range(nVar)]
    reParts = [uF.reshape((nT, nX * nY * nZ)).real ** 2 for uF in ffts]
    imParts = [uF.reshape((nT, nX * nY * nZ)).imag ** 2 for uF in ffts]

    # Spectrum computation
    if verbose:
        print(" -- computing spectrum ...")
    spectrum = np.zeros((nT, size))
    for i in idxList:
        if verbose:
            print(f" -- k{i+1}/{len(idxList)}")
        kIdx = np.argwhere(flatIdx == i)
        tmp = np.empty((nT, *kIdx.shape))
        for re, im in zip(reParts, imParts):
            np.copyto(tmp, re[:, kIdx])
            tmp += im[:, kIdx]
            spectrum[:, i] += tmp.sum(axis=(1, 2))
    spectrum /= 2 * (nX * nY * nZ) ** 2

    if verbose:
        print(" -- done !")
    return spectrum


if __name__ == '__main__':
    from pySDC.helpers.fieldsIO import FieldsIO

    data = FieldsIO.fromFile('./data/RBC3DBenchmarkSDC-res32.pySDC')
    x_coords = data.header['coords'][0]
    z_coords = data.header['coords'][-1]
    spectrum_all = []
    for i in range(len(data.times)):
        _data = data.readField(i)[1]
        vel = np.empty((1, data.dim, *data.gridSizes), data.dtype)
        vel[0][0] = _data[1]
        vel[0][1] = _data[2]
        vel[0][2] = _data[3]
        spectrum_all.append(computeMeanSpectrum(vel, x_coords, z_coords, True))
    spectrum = np.array(spectrum_all).mean(axis=(0, 1))

    import matplotlib.pyplot as plt

    plt.loglog(np.arange(len(spectrum)) + 1, spectrum)
    plt.savefig('plots/spectrum_3D.png')
