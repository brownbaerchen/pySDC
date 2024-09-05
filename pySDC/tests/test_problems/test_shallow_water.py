import pytest


@pytest.mark.base
def test_SWE_linearized(plotting=False):
    from pySDC.implementations.problem_classes.ShallowWater import ShallowWaterLinearized
    import numpy as np

    P = ShallowWaterLinearized(
        k=0, f=0, g=1e-2, H=1e2, nx=2**5 + 1, ny=2**4, left_preconditioner=False, right_preconditioning='T2T'
    )

    def implicit_euler(u, dt):
        return P.solve_system(u, dt)

    dt = 1.0e-1
    u0 = P.u_exact()

    un = implicit_euler(u0, dt)
    u_backward = un - dt * P.eval_f(un)
    # assert np.allclose(u0, u_backward)

    if not plotting:
        return None

    un = u0
    fig = P.get_fig()
    import matplotlib.pyplot as plt

    for i in range(99):
        t = dt * (i + 1)
        un = implicit_euler(un, dt)
        u_e = P.u_exact(t)
        print(f'error at {t=:.2f}: {abs(un[1]-u_e[1]):.2e}')
        # un = u_e
        P.plot(un, fig=fig, comp='h', t=t)
        plt.pause(1e-1)
    plt.show()


if __name__ == '__main__':
    test_SWE_linearized(True)