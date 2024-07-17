import pytest


@pytest.mark.base
@pytest.mark.parametrize('mode', ['T2T', 'T2U'])
def test_Burgers_f(mode):
    import numpy as np
    from pySDC.implementations.problem_classes.Burgers import Burgers1D

    P = Burgers1D(N=2**4, epsilon=8e-3, mode=mode)

    u = P.u_init
    u[0] = np.sin(P.x * np.pi)
    u[1] = np.cos(P.x * np.pi) * np.pi
    f = P.eval_f(u)

    assert np.allclose(f.impl[0], np.sin(P.x * np.pi) * P.epsilon * np.pi**2)
    assert np.allclose(f.expl[0], u[0] * u[1])
    assert np.allclose(f.impl[1], 0)
    assert np.allclose(f.expl[1], 0)


@pytest.mark.base
@pytest.mark.parametrize('mode', ['T2T', 'T2U'])
def test_Burgers_solver(mode, N=2**8, plotting=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from pySDC.implementations.problem_classes.Burgers import Burgers1D

    P = Burgers1D(N=N, epsilon=1e-2, f=0, BCl=-1, BCr=1, mode=mode)

    u = P.u_exact()
    f = P.eval_f(u)

    def imex_euler(u, f, dt):
        u = P.solve_system(u + dt * f.expl, dt)
        f = P.eval_f(u)
        return u, f

    small_step_size = 1e-9
    small_step, _ = imex_euler(u.copy(), f.copy(), small_step_size)

    assert np.allclose(u[0], small_step[0], atol=small_step_size * 1e1)

    u_exact_steady = P.u_exact(np.inf)
    dt = 1e-2
    tol = 1e-4
    for i in range(900):
        u_old = u.copy()
        u, f = imex_euler(u, f, dt)

        if plotting:
            plt.plot(P.x, u[0])
            plt.plot(P.x, u_exact_steady[0])
            plt.title(f't={i*dt:.2e}')
            plt.pause(1e-8)
            plt.cla()

        if abs(u_old[0] - u[0]) < tol:
            print(f'stopping after {i} steps')
            break

    assert np.allclose(u[0], u_exact_steady[0], atol=tol * 1e1)


if __name__ == '__main__':
    test_Burgers_solver('T2U', N=2**10, plotting=True)
    print('done')
