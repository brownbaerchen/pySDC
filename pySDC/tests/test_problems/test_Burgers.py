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


# @pytest.mark.base
# @pytest.mark.parametrize('mode', ['T2T', 'T2U'])
# def test_Burgers_solver(mode):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from pySDC.implementations.problem_classes.Burgers import Burgers1D
#
#     P = Burgers1D(N=2**10, BCl=0, BCr=0, epsilon=1e-2)
#     P = Burgers1D(N=2**8, epsilon=8e-3, f=2, BCl=-1, BCr=1)
#
#     u0 = P.u_exact()
#     f0 = P.eval_f(u0)
#     # assert np.allclose(f0.impl[0], 0)
#     # assert np.allclose(f0.expl[0], u0[0]*u0[1])
#     # plt.plot(P.x, u0[0])
#     # plt.plot(P.x, u0[1])
#     # plt.show()
#     # return 0
#
#     u = P.u_init
#     u[0] = np.sin(P.x*np.pi)
#     u[1] = np.cos(P.x*np.pi)*np.pi
#     # assert np.allclose(f.expl[0], u[0]*u[1])
#     # assert np.allclose(f.impl[0], np.sin(P.x*np.pi)*P.epsilon*np.pi**2)
#
#     u = P.u_exact()
#     f = P.eval_f(u)
#
#     dt = 1e-2
#     def imex_euler(u, f, dt):
#         u = P.solve_system(u + dt * f.expl, dt)
#         f = P.eval_f(u)
#         return u, f
#
#     for i in range(900):
#         u, f = imex_euler(u, f, dt)
#         plt.plot(P.x, u[0])
#         plt.pause(1e-8)
#         plt.cla()
#
#     # un = P.solve_system(u + dt*f.expl, dt)
#
#     plt.plot(P.x, u[0])
#     # plt.plot(P.x, un[0])
#     plt.show()

if __name__ == '__main__':
    test_Burgers_f('T2U')
