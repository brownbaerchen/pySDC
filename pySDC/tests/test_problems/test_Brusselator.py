import pytest


@pytest.mark.mpi4py
def test_Brusselator():
    """
    Test the implementation of the 2D Brusselator by doing an IMEX Euler step forward and then an explicit Euler step
    backward to compute something akin to an error. We check that the "local error" has order 2.
    """
    from pySDC.implementations.problem_classes.Brusselator import Brusselator, BrusselatorFD
    import numpy as np

    prob = Brusselator()

    dts = np.logspace(-4, -7, 15)
    errors = []
    errors_scipy = []

    for dt in dts:

        u0 = prob.u_exact(0)
        f0 = prob.eval_f(u0, 0)

        # do an IMEX Euler step forward
        u1 = prob.solve_system(u0 + dt * f0.expl, dt, u0, 0)
        u1_scipy = prob.u_exact(dt)
        errors_scipy += [abs(u1 - u1_scipy)]

        # do an explicit Euler step backward
        f1 = prob.eval_f(u1, dt)
        u02 = u1 - dt * (f1.impl + f1.expl)
        errors += [abs(u0 - u02)]

    for err, name in zip([errors, errors_scipy], ['Euler', 'scipy']):
        err = np.array(err)
        dts = np.array(dts)
        order = np.log(err[1:] / err[:-1]) / np.log(dts[1:] / dts[:-1])

        print(name, err, order)
        assert np.isclose(
            np.median(order), 2, atol=6e-2
        ), f'Expected order 2, but got {np.median(order):.2f} in {name!r} test!'


@pytest.mark.mpi4py
def test_Brusselator_fully_implicit():
    from pySDC.implementations.problem_classes.Brusselator import Brusselator, BrusselatorFD
    import numpy as np

    prob = BrusselatorFD(imex=True)

    dts = np.logspace(-3, -7, 15)
    errors = []

    for dt in dts:

        u0 = prob.u_exact(0)
        f0 = prob.eval_f(u0, 0)

        # do an implicit Euler step forward
        u1 = prob.solve_system(u0, dt, u0, 0)

        u1_scipy = prob.u_exact(dt)

        # do an explicit Euler step backward
        f1 = prob.eval_f(u1, dt)
        u02 = u1 - dt * f1
        errors += [abs(u0 - u02)]

    errors = np.array(errors)
    dts = np.array(dts)
    order = np.log(errors[1:] / errors[:-1]) / np.log(dts[1:] / dts[:-1])

    print(errors, order)
    assert np.isclose(np.median(order), 2, atol=6e-2), f'Expected order 2, but got {np.median(order):.2f}!'


if __name__ == '__main__':
    test_Brusselator()
