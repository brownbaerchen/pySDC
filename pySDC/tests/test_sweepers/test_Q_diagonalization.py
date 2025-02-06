import pytest


def get_composite_collocation_problem(L, M, N):
    import numpy as np
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.Q_diagonalization import QDiagonalization

    level_params = {}
    level_params['dt'] = 1e-3
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


if __name__ == '__main__':
    test_direct_solve(3, 2)
    # test_ParaDiag(1, 1, 1, 1e-4)
