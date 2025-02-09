import pytest


def get_composite_collocation_problem(L, M, N, alpha=0, linear=True):
    import numpy as np
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI
    from pySDC.implementations.sweeper_classes.Q_diagonalization import QDiagonalization

    level_params = {}
    level_params['dt'] = 1e-1
    level_params['restol'] = -1

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = M
    sweeper_params['initial_guess'] = 'copy'
    sweeper_params['update_f_evals'] = True

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
    controller = controller_ParaDiag_nonMPI(**controller_args, num_procs=L, alpha=alpha, linear=linear)
    P = controller.MS[0].levels[0].prob

    return controller, P, description


@pytest.mark.base
@pytest.mark.parametrize('M', [1, 3])
@pytest.mark.parametrize('N', [1, 2])
def test_direct_solve(M, N):
    """
    Test that the diagonalization has the same result as a direct solve of the collocation problem
    """
    import numpy as np
    import scipy.sparse as sp

    controller, prob, _ = get_composite_collocation_problem(1, M, N)

    controller.MS[0].levels[0].status.unlocked = True
    level = controller.MS[0].levels[0]
    level.status.time = 0
    sweep = level.sweep

    # initial conditions
    for m in range(M + 1):
        level.u[m] = prob.u_init
        level.f[m] = prob.eval_f(level.u[m], 0)
    level.u[0][:] = 1

    sweep.update_nodes()

    # solve directly
    I_MN = sp.eye((M) * N)
    Q = sweep.coll.Qmat[1:, 1:]
    C_coll = I_MN - level.dt * sp.kron(Q, prob.A)

    u0 = np.ones(shape=(M, N))
    u = sp.linalg.spsolve(C_coll, u0.flatten()).reshape(u0.shape)

    for m in range(M):
        assert np.allclose(u[m], level.u[m + 1])

    sweep.compute_residual()
    assert np.isclose(level.status.residual, 0), 'residual is non-zero'


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

    controller, prob, description = get_composite_collocation_problem(L, M, N, alpha)
    level = controller.MS[0].levels[0]

    restol = 1e-7
    dt = level.params.dt

    # setup initial conditions
    u0 = prob.u_init
    u0[:] = 1
    controller.ParaDiag_block_u0 = u0

    # initialize solution, right hand side variables and time
    for l in range(L):
        controller.MS[l].levels[0].status.time = l * dt
        for m in range(M + 1):
            controller.MS[l].levels[0].u[m] = prob.u_init
            controller.MS[l].levels[0].f[m] = prob.f_init

    controller.MS[0].levels[0].u[0][:] = u0[:]

    # do ParaDiag iterations
    res = 1
    n_iter = 0
    maxiter = 10
    while res > restol:
        controller.it_ParaDiag()

        # compute composite collocation problem residual to determine convergence (requires MPI p2p and Reduce communication)
        res = controller.compute_ParaDiag_block_residual()
        n_iter += 1
        assert n_iter < maxiter, f'Did not converge within {maxiter} iterations! Residual: {res:.2e}'
    print(f'Needed {n_iter} ParaDiag iterations, stopped at residual {res:.2e}')


if __name__ == '__main__':
    test_direct_solve(2, 1)
    test_ParaDiag(2, 2, 1, 1e-4)
