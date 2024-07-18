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


@pytest.mark.base
@pytest.mark.parametrize('mode', ['T2T', 'T2U'])
@pytest.mark.parametrize('direction', ['x', 'z', 'mixed'])
def test_Burgers2D_f(mode, direction, plotting=False):
    import numpy as np
    from pySDC.implementations.problem_classes.Burgers import Burgers2D

    nx = 2**7
    nz = 2**7
    P = Burgers2D(nx=nx, nz=nz, epsilon=8e-3, mode=mode)

    u = P.u_init
    iu, iv, iux, ivz = (P.helper.index(comp) for comp in P.helper.components)

    f_expect = P.f_init

    if direction == 'x':
        u[iu] = np.sin(P.X * 2)
        u[iux] = np.cos(P.X * 2) * 2
        f_expect.impl[iu] = -P.epsilon * u[iu] * 2**2
    elif direction == 'z':
        u[iv] = P.Z**3 + 2
        u[ivz] = P.Z**2 * 3
        f_expect.impl[iv] = -P.epsilon * P.Z * 6
    elif direction == 'mixed':
        u[iu] = np.sin(P.X * 2) * (P.Z**3 + 2)
        u[iv] = np.sin(P.X * 2) * (P.Z**3 + 2)
        u[iux] = np.cos(P.X * 2) * 2 * (P.Z**3 + 2)
        u[ivz] = np.sin(P.X * 2) * P.Z**2 * 3
        f_expect.impl[iu] = -P.epsilon * np.sin(P.X * 2) * 2**2 * (P.Z**3 + 2)
        f_expect.impl[iv] = -P.epsilon * np.sin(P.X * 2) * P.Z * 6

    f = P.eval_f(u)
    f_expect.expl[iu] = u[iu] * u[iux]
    f_expect.expl[iv] = u[iv] * u[ivz]

    if plotting:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2)
        i = iv
        axs[0].pcolormesh(P.X, P.Z, f_expect.impl[i].real)
        axs[1].pcolormesh(P.X, P.Z, (f.impl[i]).real)
        plt.show()

    for comp in P.helper.components:
        i = P.helper.index(comp)
        assert np.allclose(f.expl[i], f_expect.expl[i]), f'Error in component {comp}!'
        assert np.allclose(f.impl[i], f_expect.impl[i]), f'Error in component {comp}!'


if __name__ == '__main__':
    # test_Burgers_solver('T2U', plotting=True)
    test_Burgers2D_f('T2T', 'mixed')
