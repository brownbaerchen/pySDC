import numpy as np
import pytest

from pySDC.core.sweeper import Sweeper as Sweeper

node_types = ['EQUID', 'LEGENDRE'] + [f'CHEBY-{i}' for i in [1, 2, 3, 4]]
quad_types = ['GAUSS', 'LOBATTO', 'RADAU-RIGHT', 'RADAU-LEFT']
num_nodes = [2, 3, 4, 5]


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_MIN_SR(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params)
    Q = sweeper.coll.Qmat[1:, 1:]

    # Check non-stiff limit
    QDelta = sweeper.get_Qdelta_implicit('MIN-SR-NS')[1:, 1:]
    assert np.all(np.diag(np.diag(QDelta)) == QDelta), "no diagonal QDelta"
    K = Q - QDelta
    Km = np.linalg.matrix_power(K, M)
    nilpotency = np.linalg.norm(Km, ord=np.inf)
    assert nilpotency < 1e-10, "Q-QDelta not nilpotent " f"(M={M}, norm={nilpotency})"

    # Check stiff limit
    QDelta = sweeper.get_Qdelta_implicit('MIN-SR-S')[1:, 1:]
    assert np.all(np.diag(np.diag(QDelta)) == QDelta), "no diagonal QDelta"

    if params['quad_type'] in ['LOBATTO', 'RADAU-LEFT']:
        QDelta = np.diag(1 / np.diag(QDelta[1:, 1:]))
        Q = Q[1:, 1:]
    else:
        QDelta = np.diag(1 / np.diag(QDelta))

    K = np.eye(Q.shape[0]) - QDelta @ Q
    Km = np.linalg.matrix_power(K, M)
    nilpotency = np.linalg.norm(Km, ord=np.inf)
    assert nilpotency < 1e-10, "I-QDelta^{-1}Q not nilpotent " f"(M={M}, norm={nilpotency})"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_LU(node_type, quad_type, M):
    if M > 3 and node_type == 'EQUID' and quad_type in ['GAUSS', 'RADAU-RIGHT']:
        # Edge case for some specific equidistant nodes
        # TODO : still need to be understood ...
        return

    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params)
    Q = sweeper.coll.Qmat[1:, 1:]

    # Check nilpotency
    QDelta = sweeper.get_Qdelta_implicit('LU')[1:, 1:]

    if params['quad_type'] in ['LOBATTO', 'RADAU-LEFT']:
        QDelta = QDelta[1:, 1:]
        Q = Q[1:, 1:]

    K = np.eye(Q.shape[0]) - np.linalg.solve(QDelta, Q)
    Km = np.linalg.matrix_power(K, M)
    nilpotency = np.linalg.norm(Km, ord=np.inf)
    assert nilpotency < 1e-14, "I-QDelta^{-1}Q not nilpotent " f"(M={M}, norm={nilpotency})"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_Qpar(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params)
    Q = sweeper.coll.Qmat[1:, 1:]

    QDelta = sweeper.get_Qdelta_implicit('Qpar')[1:, 1:]
    assert np.all(np.diag(np.diag(QDelta)) == QDelta), "no diagonal QDelta"
    assert np.all(np.diag(QDelta) == np.diag(Q)), "not the diagonal Q coefficients"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_IE(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params)

    QDelta = sweeper.get_Qdelta_implicit('IE')[1:, 1:]
    for i in range(M):
        assert np.all(QDelta[i, : i + 1] == QDelta[-1, : i + 1]), "not the same coefficients in columns"
    assert np.all(np.cumsum(QDelta[-1] == sweeper.coll.nodes)), "last line cumsum not equal to nodes"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_IEpar(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params)

    QDelta = sweeper.get_Qdelta_implicit('IEpar')[1:, 1:]
    assert np.all(np.diag(np.diag(QDelta)) == QDelta), "no diagonal QDelta"
    assert np.all(np.cumsum(np.diag(QDelta) == sweeper.coll.nodes)), "diagonal cumsum not equal to nodes"


@pytest.mark.base
@pytest.mark.parametrize("node_type", node_types)
@pytest.mark.parametrize("quad_type", quad_types)
@pytest.mark.parametrize("M", num_nodes)
def test_PIC(node_type, quad_type, M):
    params = {'num_nodes': M, 'quad_type': quad_type, 'node_type': node_type}
    sweeper = Sweeper(params)

    QDelta = sweeper.get_Qdelta_implicit('PIC')[1:, 1:]
    assert np.all(QDelta == 0), "not a null matrix"


@pytest.mark.base
@pytest.mark.parametrize("M", num_nodes)
@pytest.mark.parametrize("k", num_nodes)
def test_MIN_SR_FLEX(M, k):
    if k > M:
        return None

    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d

    description = {}
    description['level_params'] = {'dt': 1.0, 'restol': -1}
    description['sweeper_params'] = {'num_nodes': M, 'QI': 'MIN-SR-FLEX', 'quad_type': 'RADAU-RIGHT'}
    description['step_params'] = {
        'maxiter': k,
    }
    description['sweeper_class'] = generic_implicit
    description['problem_class'] = testequation0d
    description['problem_params'] = {'lambdas': [[-1]]}

    controller_params = {}
    controller = controller_nonMPI(1, controller_params, description)
    u0 = controller.MS[0].levels[0].prob.u_exact(0)

    QI_before = controller.MS[0].levels[0].sweep.QI.copy()
    controller.run(u0=u0, Tend=1, t0=0)
    QI_after = controller.MS[0].levels[0].sweep.QI.copy()

    assert np.allclose([QI_before[i, i] / QI_after[i, i] for i in range(1, M + 1)], k)


if __name__ == '__main__':
    test_MIN_SR_FLEX(2, 2)
    test_MIN_SR('LEGENDRE', 'RADAU-RIGHT', 4)
    test_MIN_SR('EQUID', 'LOBATTO', 5)

    test_LU('LEGENDRE', 'RADAU-RIGHT', 4)
    test_LU('EQUID', 'LOBATTO', 5)
