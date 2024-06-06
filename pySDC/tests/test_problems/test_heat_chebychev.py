import pytest


@pytest.mark.base
def test_heat1d_chebychev(plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import Heat1DChebychev

    P = Heat1DChebychev(nvars=8, poly_coeffs=[10, 0, 0, 1, 0])

    u0 = P.u_exact()
    dt = 1e-1
    sol = P.solve_system(rhs=u0, factor=dt)
    backward = sol - dt * P.eval_f(sol)
    forward = u0 + dt * P.eval_f(u0)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(P.x, u0, label='u0')
        plt.plot(P.x, sol, ls=':', label='BE')
        plt.plot(P.x, backward, ls='-.', label='BFE')
        # plt.plot(P.x, forward, ls='-.', label='FE')
        plt.legend(frameon=False)
        plt.show()

    assert np.allclose(u0, P.solve_system(u0, 0, u0))
    assert np.allclose(u0, backward, atol=1e-7), abs(u0 - backward)


@pytest.mark.base
def test_heat2d(plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import Heat2d

    P = Heat2d()

    u0 = P.u_exact()

    if plot:
        import matplotlib.pyplot as plt

        plt.pcolormesh(P.X, P.Z, u0)
        plt.show()


if __name__ == '__main__':
    test_heat1d_chebychev(plot=True)
