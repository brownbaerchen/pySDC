import numpy as np
import scipy.sparse as sp

from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

# setup parameters
L = 1
M = 1
N = 1
alpha = 1  # e-4
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

E = sp.diags(
    [
        -1.0,
    ]
    * (L - 1),
    offsets=-1,
)
H_M = sp.eye(M).tocsc() * 0
H_M[:, -1] = 1

C_coll = I_MN - dt * sp.kron(sweep.coll.Qmat[1:, 1:], prob.A)
H = sp.kron(H_M, I_N)

C = sp.kron(I_L, C_coll) + sp.kron(E, H)

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
).tocsc()
E_alpha[0, -1] = -alpha

D_alpha = sp.diags([alpha ** (1 / L) * np.exp(-2 * np.pi * 1j * l / L) for l in range(L)])
C_alpha = sp.kron(E_alpha, H) + sp.kron(I_L, C_coll)
C_alpha_diag = sp.kron(D_alpha, H) + sp.kron(I_L, C_coll)

J = sp.kron(sp.diags([alpha ** (-l / L) for l in range(L)]), I_MN)
J_inv = sp.kron(sp.diags([1 / alpha ** (-l / L) for l in range(L)]), I_MN)

# ParaDiag
sol_paradiag = u.copy()
u0 = u.copy()


def residual(_u):
    return np.linalg.norm(C @ _u.flatten() - u.flatten(), np.inf)


res = residual(sol_paradiag)

print(sol_paradiag)
while res > restol:
    x = np.fft.ifft(
        (J_inv @ ((C_alpha - C) @ sol_paradiag.flatten() + u0.flatten())).reshape(sol_paradiag.shape), axis=0
    )
    y = sp.linalg.spsolve(C_alpha_diag, x.flatten()).reshape(x.shape)
    sol_paradiag = (J @ np.fft.fft(y, axis=0).flatten()).reshape(y.shape)
    res = residual(sol_paradiag)
    print(sol_paradiag)
    print(res)
    breakpoint()

# print(D_alpha.toarray())


breakpoint()
