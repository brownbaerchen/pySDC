import pytest


@pytest.mark.firedrake
def test_addition(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = firedrake_mesh(V)
    b = firedrake_mesh(a)

    # TODO: get rid of this, but somehow, I need this to get `.dat._numpy_data`.
    x = fd.SpatialCoordinate(mesh)
    ic = fd.project(fd.as_vector([fd.sin(fd.pi * x[0]), 0]), V)
    a.assign(ic)
    b.assign(ic)

    a.dat._numpy_data[:] = v1

    b.dat._numpy_data[:] = v2

    c = a + b

    assert np.allclose(c.dat._numpy_data, v1 + v2)
    assert np.allclose(a.dat._numpy_data, v1)
    assert np.allclose(b.dat._numpy_data, v2)


@pytest.mark.firedrake
def test_subtraction(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = firedrake_mesh(V)
    b = firedrake_mesh(a)

    # TODO: get rid of this, but somehow, I need this to get `.dat._numpy_data`.
    x = fd.SpatialCoordinate(mesh)
    ic = fd.project(fd.as_vector([fd.sin(fd.pi * x[0]), 0]), V)
    a.assign(ic)
    b.assign(ic)

    a.dat._numpy_data[:] = v1

    b.dat._numpy_data[:] = v2

    c = a - b

    assert np.allclose(c.dat._numpy_data, v1 - v2)
    assert np.allclose(a.dat._numpy_data, v1)
    assert np.allclose(b.dat._numpy_data, v2)


@pytest.mark.firedrake
def test_right_multiplication(n=3, v1=1, v2=2):
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 2)

    a = firedrake_mesh(V)
    b = firedrake_mesh(a)

    # TODO: get rid of this, but somehow, I need this to get `.dat._numpy_data`.
    x = fd.SpatialCoordinate(mesh)
    ic = fd.project(fd.as_vector([fd.sin(fd.pi * x[0]), 0]), V)
    a.assign(ic)

    a.dat._numpy_data[:] = v1

    b = v2 * a

    assert np.allclose(b.dat._numpy_data, v1 * v2)
    assert np.allclose(a.dat._numpy_data, v1)


@pytest.mark.firedrake
def test_norm(n=3, v1=-1):
    from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh
    import numpy as np
    import firedrake as fd

    mesh = fd.UnitSquareMesh(n, n)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)

    a = firedrake_mesh(V)
    b = firedrake_mesh(a)

    # TODO: get rid of this, but somehow, I need this to get `.dat._numpy_data`.
    x = fd.SpatialCoordinate(mesh)
    ic = fd.project(fd.as_vector([fd.sin(fd.pi * x[0]), 0]), V)
    a.assign(ic)

    a.dat._numpy_data[:] = v1

    b = abs(a)

    assert np.isclose(b, np.sqrt(2) * abs(v1)), f'{b=}, {v1=}'
    assert np.allclose(a.dat._numpy_data, v1)


if __name__ == '__main__':
    test_norm(1, -2)
