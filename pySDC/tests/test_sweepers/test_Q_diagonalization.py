import pytest


def get_composite_collocation_problem(L, M, N):
    import numpy as np
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.Q_diagonalization import QDiagonalization

    level_params = {}
    level_params['dt'] = 1e-1
    level_params['restol'] = -1

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = M
    sweeper_params['initial_guess'] = 'copy'

    step_params = {}
    step_params['maxiter'] = 1

    problem_params = {'lambdas': -1.0 * np.ones(shape=(N))}

    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = []
    controller_params['mssdc_jac'] = False

    description = {}
    description['problem_class'] = testequation0d
    description['problem_params'] = problem_params
    description['sweeper_class'] = QDiagonalization
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }
    controller = controller_nonMPI(**controller_args, num_procs=L)
    P = controller.MS[0].levels[0].prob

    return controller, P


@pytest.mark.base
@pytest.mark.parametrize('M', [1, 3])
@pytest.mark.parametrize('N', [1, 2])
def test_direct_solve(M, N):
    """
    Test that the diagonalization has the same result as a direct solve of the collocation problem
    """
    import numpy as np
    import scipy.sparse as sp

    controller, prob = get_composite_collocation_problem(1, M, N)

    controller.MS[0].levels[0].status.unlocked = True
    level = controller.MS[0].levels[0]
    level.status.time = 0
    sweep = level.sweep

    # initial conditions
    for m in range(M + 1):
        level.u[m] = prob.u_init
        level.u[m][:] = 1
        level.f[m] = prob.eval_f(level.u[m], 0)

    sweep.update_nodes()

    # solve directly
    I_MN = sp.eye((M) * N)
    Q = sweep.coll.Qmat[1:, 1:]
    C_coll = I_MN - level.dt * sp.kron(Q, prob.A)

    u0 = np.ones(shape=(M, N))
    u0[0] = 1
    u = sp.linalg.spsolve(C_coll, u0.flatten()).reshape(u0.shape)

    for m in range(M):
        assert np.allclose(u[m], level.u[m + 1])

    integral = sweep.integrate()
    residual = [abs(level.u[m + 1] - integral[m] - level.u[0]) for m in range(M)]
    assert np.allclose(residual, 0), 'residual is non-zero'


@pytest.mark.base
@pytest.mark.parametrize('L', [1, 4])
@pytest.mark.parametrize('M', [2, 3])
@pytest.mark.parametrize('N', [1, 2])
@pytest.mark.parametrize('alpha', [1e-4, 1e-2])
def test_ParaDiag(L, M, N, alpha):
    import numpy as np
    import scipy.sparse as sp
    from pySDC.implementations.sweeper_classes.Q_diagonalization import QDiagonalization
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.helpers.ParaDiagHelper import (
        get_FFT_matrix,
        get_E_matrix,
        get_weighted_FFT_matrix,
        get_weighted_iFFT_matrix,
    )

    controller, prob = get_composite_collocation_problem(L, M, N)
    level = controller.MS[0].levels[0]
    sweep = level.sweep

    restol = 1e-7
    dt = level.params.dt

    # setup infrastructure
    u = np.zeros((L, M, N))

    I_M = sp.eye(M)

    E = get_E_matrix(L, 0)
    E_alpha = get_E_matrix(L, alpha)
    H_M = sweep.get_H_matrix()

    Q = sweep.coll.Qmat[1:, 1:]

    gamma = alpha ** (-np.arange(L) / L)
    diags = np.fft.fft(1 / gamma * E_alpha[:, 0].toarray().flatten(), norm='backward')
    G = [(diags[l] * H_M + I_M).tocsc() for l in range(L)]
    G_inv = [sp.linalg.inv(_G).toarray() for _G in G]

    # get the G_inv matrices into the sweepers
    for l in range(L):
        w, S, S_inv = sweep.computeDiagonalization(sweep.coll.Qmat[1:, 1:] @ G_inv[l])
        controller.MS[l].levels[0].sweep.w = w
        controller.MS[l].levels[0].sweep.S = S
        controller.MS[l].levels[0].sweep.S_inv = S_inv
        controller.MS[l].levels[0].sweep.params.G_inv = G_inv[l]

    weighted_FFT = get_weighted_FFT_matrix(L, alpha)
    weighted_iFFT = get_weighted_iFFT_matrix(L, alpha)

    def mat_vec(mat, vec):
        res = np.zeros_like(vec).astype(complex)
        for l in range(mat.shape[0]):
            for k in range(mat.shape[1]):
                res[l] += mat[l, k] * vec[k]
        return res

    def residual(_u, u0):
        res = []
        for l in range(_u.shape[0]):
            f_evals = np.array([prob.eval_f(_u[l, k], 0) for k in range(_u.shape[1])])
            Qf = mat_vec(Q, f_evals)

            _res = [_u[l][k] - dt * Qf[k] for k in range(_u.shape[1])]
            for k in range(_u.shape[1]):
                if l == 0:
                    _res[k] -= u0[0][k]
                else:
                    _res[k] -= _u[l - 1][k]
            res.append(np.max(_res))

        return np.linalg.norm(res)

    sol_paradiag = u.copy()
    u0 = u.copy()
    u0[0] = 1
    n_iter = 0

    res = residual(sol_paradiag, u0)

    while res > restol:
        # prepare local RHS to be transformed
        Hu = np.empty_like(sol_paradiag)
        for l in range(L):
            Hu[l] = H_M @ sol_paradiag[l]

        # assemble right hand side from LxL matrices and local rhs
        rhs = mat_vec((E_alpha - E).tolil(), Hu)
        rhs += u0

        # weighted FFT in time
        x = mat_vec(weighted_FFT, rhs)

        # perform local solves of "collocation problems" on the steps in parallel
        y = np.empty_like(x)
        for l in range(L):
            controller.MS[l].levels[0].u[0] = x[l][0]
            for m in range(M):
                controller.MS[l].levels[0].u[m + 1] = x[l][m]
                controller.MS[l].levels[0].status.time = l * dt

            controller.MS[l].levels[0].sweep.update_nodes()

            for m in range(M):
                y[l, m, ...] = controller.MS[l].levels[0].u[m + 1]

        # inverse FFT in time
        sol_paradiag = mat_vec(weighted_iFFT, y)

        res = residual(sol_paradiag, u0)
        n_iter += 1
        maxiter = 10
        assert n_iter < maxiter, f'Did not converge within {maxiter} iterations! Residual: {res:.2e}'
    print(f'Needed {n_iter} iterations in parallel and local paradiag, stopped at residual {res:.2e}')


if __name__ == '__main__':
    # test_direct_solve(3, 2)
    test_ParaDiag(1, 2, 1, 1e-4)
