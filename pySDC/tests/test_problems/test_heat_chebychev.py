import pytest


@pytest.mark.base
def test_heat1d_chebychev(plot=False):
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import Heat1DChebychev
    import scipy
    from pySDC.helpers.problem_helper import ChebychovHelper

    N = 2**5
    P = Heat1DChebychev(nvars=N, a=-1, b=3, poly_coeffs=[0, 0, 1, -1, 1], solver_type='gmres')
    cheby = ChebychovHelper(N)

    u0 = P.u_exact()
    u_hat = scipy.fft.dct(u0, axis=1) * P.norm

    dt = 1e-2
    sol = P.solve_system(rhs=u0, factor=dt)
    sol_hat = scipy.fft.dct(sol, axis=1) * P.norm

    # for computing forward Euler, we need to reevaluate the spatial derivative
    backward = sol - dt * P.eval_f(sol)
    backward[P.idu] = P._compute_derivative(backward[P.iu])

    # k=1
    # source_term = np.sin(k * P.x * 2 * np.pi / L)
    # sol_ex = P.solve_system(rhs=source_term
    # u = spsolve(A, source_term - b)
    # u_expect = (bc_right - bc_left) * x / L + bc_left - source_term / k**2

    if plot:
        import matplotlib.pyplot as plt

        for i in [P.iu]:
            plt.plot(P.x, u0[i], label=f'u0[{i}]')
            plt.plot(P.x, sol[i], ls='--', label=f'BE[{i}]')
            plt.plot(P.x, backward[i], ls='-.', label=f'BFE[{i}]')
        plt.legend(frameon=False)
        plt.show()

    # test that the solution satisfies the algebraic constraints
    D_u = scipy.fft.idct(P.U2T @ P.D @ sol_hat[P.iu] / P.norm)
    assert np.allclose(D_u, sol[P.idu]), 'The solution of backward Euler does not satisfy the algebraic constraints'

    # # test that f evaluation is second derivative of the solution
    # f = P.eval_f(u0)
    # D2 = cheby.get_T2T_differentiation_matrix(2)
    # D1 = cheby.get_T2T_differentiation_matrix(1)
    # D2u = scipy.fft.idct(D2 @ u_hat[P.iu] / P.norm)
    # D1u = scipy.fft.idct(D1 @ u_hat[P.iu] / P.norm)
    # assert np.allclose(D2u, f[P.iu]), 'The time derivative is not the second space derivative.'
    # assert np.allclose(
    #     D1u, u0[P.idu]
    # ), 'The initial conditions don\'t have the first space derivative where it needs to be.'

    assert np.allclose(
        u0, P.solve_system(u0, 1e-9, u0)
    ), 'We did not get back the initial conditions when solving with \"zero\" step size.'
    assert np.allclose(u0, backward, atol=1e-7), abs(u0 - backward)


# def test_condition_number:
#     import numpy as np
#     from pySDC.implementations.problem_classes.HeatEquation_1D_Chebychev import Heat1DChebychev
#     import scipy
#     from pySDC.helpers.problem_helper import ChebychovHelper
#
#     N = 5
#     P = Heat1DChebychev(nvars=N, a=-1, b=3, poly_coeffs=[0, 0, 1, -1, 1])


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
