import pytest


@pytest.mark.base
def test_heat1d_chebychev(plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import Heat1DChebychev

    P = Heat1DChebychev(nvars=128, use_cheby_grid=False)

    u0 = P.u_exact()
    dt = 5e-2
    sol = P.solve_system(rhs=u0, factor=dt)
    backward = sol - dt * P.eval_f(sol)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(P.x, u0, label='u0')
        plt.plot(P.x, sol, ls=':', label='BE')
        plt.plot(P.x, backward, ls='-.', label='BFE')
        plt.legend(frameon=False)
        plt.show()

    assert np.allclose(P.y, P._interpolate_to_chebychev_grid(P.x))
    assert np.allclose(u0, P.solve_system(u0, 0, u0))
    assert np.allclose(u0, backward, atol=1e-7), abs(u0 - backward)


if __name__ == '__main__':
    test_heat1d_chebychev(plot=True)
