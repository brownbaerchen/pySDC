import numpy as np
import scipy.sparse as sp

from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol

# setup parameters
L = 1
M = 2
N = 2
alpha = 1e-4
restol = 1e-7
dt = 0.1

sweeper_params = {
    'num_nodes': M,
    'quad_type': 'RADAU-RIGHT',
}

# setup infrastructure
prob = vanderpol(newton_maxiter=1, mu=1e-0, crash_at_maxiter=False)

# make problem work on complex data
prob.init = tuple([*prob.init[:2]] + [np.dtype('complex128')])

N = prob.init[0]
u = np.zeros((L, M, N), dtype=complex)

# setup collocation problem
sweep = sweeper_class(sweeper_params)

# initial conditions
u[0, :, :] = prob.u_exact(t=0)


# setup matrices
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

J_L = sp.diags(gamma)
J_L_inv = sp.diags(1 / gamma)


def mat_vec(mat, vec):
    res = np.zeros_like(vec)
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


# ParaDiag with local solves
G = [(D_alpha.tocsc()[l, l] * H_M + I_M).tocsc() for l in range(L)]
sol_paradiag = u.copy()
u0 = u.copy()
n_iter = 0

res = residual(sol_paradiag, u0)

# prepare diagonalization of QG
w = []
S = []
S_inv = []

for l in range(L):
    # diagonalize QG^-1 matrix
    if M > 1:
        _w, _S = np.linalg.eig(Q @ sp.linalg.inv(G[l]).toarray())
    else:
        _w, _S = np.linalg.eig(Q / (G[l].toarray()))
    _S_inv = np.linalg.inv(_S)
    w.append(_w)
    S.append(_S)
    S_inv.append(_S_inv)


while res > restol:
    # prepare local RHS to be transformed
    Hu = np.empty_like(sol_paradiag)
    for l in range(L):
        Hu[l] = H_M @ sol_paradiag[l]

    # broadcast average solution to construct average Jacobians (one for each collocation node)
    _u_avg = np.zeros(shape=(M, N), dtype=complex)
    for l in range(L):
        _u_avg += S_inv[l] @ sol_paradiag[l] / L
    u_avg = [prob.u_init for _ in range(M)]
    for i in range(M):
        u_avg[i][:] = _u_avg[i]

    # compute contribution to RHS in non-linear problems
    rhs_nonlin = np.empty_like(sol_paradiag)
    for l in range(L):
        f_evals = np.array([prob.eval_f(sol_paradiag[l, m], 0) for m in range(M)])
        Qf = mat_vec(Q, f_evals)
        J_evals = np.array([prob.evaluate_jacobian(u_avg[m], sol_paradiag[l, m]) for m in range(M)])
        Qf_avg = mat_vec(Q, J_evals)
        rhs_nonlin[l] = dt * (Qf - Qf_avg)

    # assemble right hand side from LxL matrices and local rhs
    rhs = mat_vec((E_alpha - E).tolil(), Hu)
    rhs += rhs_nonlin
    rhs += u0

    # weighted FFT in time
    x = np.fft.fft(mat_vec(J_L_inv.toarray(), rhs), axis=0)

    # perform local solves of "collocation problems" on the steps in parallel
    y = np.empty_like(x)
    for l in range(L):

        # perform local solves on the collocation nodes in parallel
        x1 = S_inv[l] @ x[l]
        x2 = np.empty_like(x1)
        for m in range(M):
            x2[m, :] = prob.solve_system(x1[m], w[l][m] * dt, u0=u_avg[m], t=l * dt)
        z = S[l] @ x2
        y[l, ...] = sp.linalg.spsolve(G[l], z)

    # inverse FFT in time
    sol_paradiag = mat_vec(J_L.toarray(), np.fft.ifft(y, axis=0))

    res = residual(sol_paradiag, u0)
    n_iter += 1
    print(n_iter, res)
print(f'Needed {n_iter} ParaDiag iterations, stopped at residual {res:.2e}')
