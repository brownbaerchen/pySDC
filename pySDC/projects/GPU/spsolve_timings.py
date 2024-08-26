import scipy.sparse.linalg as lina
import scipy.sparse as sp
import numpy as xp

N = 2**4


def single_test(sp, xp, lina, N):
    A = sp.eye(N).tocsr()
    y = xp.zeros(N)
    u = lina.spsolve(A, y)


from cupyx.profiler import benchmark


def test_GPU():
    import cupy as xp
    import cupyx.scipy.sparse as sp
    import cupyx.scipy.sparse.linalg as lina

    N = 2**10

    args = (sp, xp, lina, N)
    print(benchmark(single_test, args))


if __name__ == '__main__':
    test_GPU()
