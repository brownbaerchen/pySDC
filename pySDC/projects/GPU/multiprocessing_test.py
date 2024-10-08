import multiprocessing as mp
from multiprocessing import Pool, Process

# from pathos.multiprocessing import ProcessingPool as Pool
from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard
from time import perf_counter
from scipy.sparse.linalg import factorized, splu, SuperLU, spsolve_triangular
import scipy.sparse as sp
import numpy as np
import concurrent.futures


class MySuperLU:
    def __init__(self, shape, nnz, L, U, perm_r, perm_c):
        """LU factorization of a sparse matrix.

        Args:
            obj (scipy.sparse.linalg.SuperLU): LU factorization of a sparse
                matrix, computed by `scipy.sparse.linalg.splu`, etc.
        """
        self.shape = shape
        self.nnz = nnz
        self.L = L
        self.U = U
        self.perm_r = perm_r
        self.perm_c = perm_c

    def solve(self, b):
        self.Pr = sp.csc_matrix((np.ones(self.shape[0]), (self.perm_r, np.arange(self.shape[0]))))
        self.Pl = sp.csc_matrix((np.ones(self.shape[0]), (np.arange(self.shape[0]), self.perm_c)))
        y = spsolve_triangular(self.L, self.Pr @ b)
        z = spsolve_triangular(self.U, y, lower=False)
        return self.Pl @ z

    def __call__(self, b):
        return self.solve(b)


def my_factorized(M):
    _LU = splu(M)
    return MySuperLU(_LU.shape, _LU.nnz, _LU.L, _LU.U, _LU.perm_r, _LU.perm_c)


def compare_my_stuff_vs_scipy(A, b):
    t0 = perf_counter()
    referenceLU = factorized(A)
    reference_sol = referenceLU(b)

    t1 = perf_counter()

    myLU = my_factorized(A)
    my_sol = myLU(b)

    t2 = perf_counter()

    assert np.allclose(reference_sol, my_sol)
    print(f'Comparison of scipy and my stuff passed with slowdown of {(t2-t1)/(t1-t0):.2e}', flush=True)


def single_test(num_threads, nx=512, useMP=True):
    local_size = nx // num_threads

    prob = RayleighBenard(nx=local_size, nz=nx // 4)

    rhs = tuple(
        [
            prob.u_init.flatten(),
        ]
        * num_threads
    )
    M = tuple(
        [
            prob.put_BCs_in_matrix(prob.L).tocsc(),
        ]
        * num_threads
    )

    if num_threads > 1:
        if useMP:
            pool = Pool(processes=num_threads)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=num_threads)
    else:
        pool = None

    t0 = perf_counter()
    if pool:
        solvers = pool.map(my_factorized, M)
    else:
        solvers = tuple(
            [
                my_factorized(M[0]),
            ]
        )

    t1 = perf_counter()

    print(f'With {num_threads} threads, I need {t1-t0:.2e}s to factorize the matrix', flush=True)
    # print(f'With {num_threads} threads, I need {t1-t0:.2e}s to factorize the matrix and {t2-t1:.2e}s to use the factorization', flush=True)


if __name__ == '__main__':
    mp.set_start_method('fork')
    num_threads = 1
    nx = 512
    local_size = nx // num_threads

    prob = RayleighBenard(nx=local_size, nz=nx // 4)

    rhs = tuple(
        [
            prob.u_init.flatten(),
        ]
        * num_threads
    )
    M = tuple(
        [
            prob.put_BCs_in_matrix(prob.L).tocsc(),
        ]
        * num_threads
    )

    for num_threads in [1, 2, 4, 8]:
        single_test(num_threads)

    compare_my_stuff_vs_scipy(M[0], rhs[0])
