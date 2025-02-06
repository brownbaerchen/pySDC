import numpy as np
import scipy.sparse as sp
from pySDC.helpers.ParaDiagHelper import get_fft_matrix

from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class
from pySDC.implementations.sweeper_classes.Q_diagonalization import QDiagonalization

# setup parameters
L = 4
M = 3
N = 2
alpha = 1e-4
restol = 1e-7
dt = 0.1

sweeper_params = {
    'num_nodes': M,
    'quad_type': 'RADAU-RIGHT',
}

# setup infrastructure
prob = problem_class(lambdas=-1.0 * np.ones(shape=(N)))
u = np.zeros((L, M, N))
sweep = sweeper_class(sweeper_params)

# initial conditions
u[0, :, :] = 1.0

# setup matrices for composite collocation problem
I_L = sp.eye(L)
I_MN = sp.eye((M) * N)
I_N = sp.eye(N)
I_M = sp.eye(M)

E = sp.diags(
    [
        -1.0,
    ]
    * (L - 1),
    offsets=-1,
)
H_M = sp.eye(M).tolil() * 0
H_M[:, -1] = 1

Q = sweep.coll.Qmat[1:, 1:]
C_coll = I_MN - dt * sp.kron(Q, prob.A)
H = sp.kron(H_M, I_N)

C = (sp.kron(I_L, C_coll) + sp.kron(E, H)).tocsc()

# solve composite collocation problem sequentially with forward substitution
sol_seq = u.copy()
for l in range(L):
    _sol = sp.linalg.spsolve(C_coll, sol_seq[l].flatten()).reshape(sol_seq[l].shape)
    sol_seq[l, :] = _sol
    if l < L - 1:
        sol_seq[l + 1, :] = _sol[-1, :]

# solve composite collocation problem directly
sol_direct = sp.linalg.spsolve(C, u.flatten()).reshape(u.shape)
assert np.allclose(sol_seq, sol_direct)

# setup ParaDiag
E_alpha = sp.diags(
    [
        -1.0,
    ]
    * (L - 1),
    offsets=-1,
).tolil()
E_alpha[0, -1] = -alpha

gamma = alpha ** (-np.arange(L) / L)
D_alpha = sp.diags(np.fft.fft(1 / gamma * E_alpha[:, 0].toarray().flatten(), norm='backward'))
C_alpha = (sp.kron(E_alpha, H) + sp.kron(I_L, C_coll)).tocsc()
C_alpha_diag = (sp.kron(D_alpha, H) + sp.kron(I_L, C_coll)).tocsc()

J = sp.kron(sp.diags(gamma), I_MN)
J_inv = sp.kron(sp.diags(1 / gamma), I_MN)
J_L = sp.diags(gamma)
J_L_inv = sp.diags(1 / gamma)


def mat_vec(mat, vec):
    res = np.zeros_like(vec).astype(complex)
    for l in range(vec.shape[0]):
        for k in range(vec.shape[0]):
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


fft_mat = get_fft_matrix(L)


# ParaDiag without diagonalization and FFTs
sol_paradiag_serial = u.copy()
u0 = u.copy()
res_serial = residual(sol_paradiag_serial, u0)
n_iter_serial = 0

while res_serial > restol:
    sol_paradiag_serial = sp.linalg.spsolve(
        C_alpha, (C_alpha - C) @ sol_paradiag_serial.flatten() + u0.flatten()
    ).reshape(sol_paradiag_serial.shape)
    res_serial = residual(sol_paradiag_serial, u0)
    n_iter_serial += 1
print(f'Needed {n_iter_serial} iterations in serial paradiag, stopped at residual {res_serial:.2e}')
assert np.allclose(sol_paradiag_serial, sol_direct)

# ParaDiag
sol_paradiag_Kron = u.copy()
u0 = u.copy()
n_iter_Kron = 0

res = residual(sol_paradiag_Kron, u0)

while res > restol:
    x = np.fft.fft(
        (J_inv @ ((C_alpha - C) @ sol_paradiag_Kron.flatten() + u0.flatten())).reshape(sol_paradiag_Kron.shape),
        axis=0,
        norm='ortho',
    )
    y = sp.linalg.spsolve(C_alpha_diag, x.flatten()).reshape(x.shape)
    sol_paradiag_Kron = (J @ np.fft.ifft(y, axis=0, norm='ortho').flatten()).reshape(y.shape)

    res = residual(sol_paradiag_Kron, u0)
    n_iter_Kron += 1
print(f'Needed {n_iter_Kron} iterations in parallel paradiag, stopped at residual {res:.2e}')
assert np.allclose(sol_paradiag_Kron, sol_direct)
assert np.allclose(n_iter_Kron, n_iter_serial)


# ParaDiag with local solves
G = [(D_alpha.tocsc()[l, l] * H_M + I_M).tocsc() for l in range(L)]
G_inv = [sp.linalg.inv(_G).toarray() for _G in G]
sol_paradiag = u.copy()
u0 = u.copy()
n_iter = 0
sweepers = [QDiagonalization(params={**sweeper_params, 'G_inv': _G_inv}) for _G_inv in G_inv]

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
    x = mat_vec(fft_mat, mat_vec(J_L_inv.toarray(), rhs))

    # perform local solves of "collocation problems" on the steps in parallel
    y = np.empty_like(x)
    for l in range(L):

        # diagonalize QG^-1 matrix
        w, S, S_inv = sweepers[l].w, sweepers[l].S, sweepers[l].S_inv

        # perform local solves on the collocation nodes in parallel
        x1 = S_inv @ x[l]
        x2 = np.empty_like(x1)
        for m in range(M):
            x2[m, :] = prob.solve_system(rhs=x1[m], factor=w[m] * dt, u0=x1[m], t=0)
        z = S @ x2
        y[l, ...] = G_inv[l] @ z

    # inverse FFT in time
    sol_paradiag = mat_vec(J_L.toarray(), mat_vec(np.conjugate(fft_mat), y))

    res = residual(sol_paradiag, u0)
    n_iter += 1
print(f'Needed {n_iter} iterations in parallel and local paradiag, stopped at residual {res:.2e}')
assert np.allclose(sol_paradiag, sol_direct)
assert np.allclose(n_iter, n_iter_serial)
