from mpi4py import MPI
from pySDC.implementations.sweeper_classes.generic_implicit_MPI import SweeperMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order


class imex_1st_order_MPI(SweeperMPI, imex_1st_order):
    def __init__(self, params):
        super().__init__(params)
        assert (
            self.params.QE == 'PIC'
        ), f"Only Picard is implemented for explicit precondioner so far in {type(self).__name__}! You chose \"{self.params.QE}\""

    def add_Q_minus_QD_times_F(self, rhs):
        self.add_implicit_explicit_matrix_times_f_evaluations_to(
            self.coll.Qmat - self.QI, self.coll.Qmat - self.QE, rhs
        )

    def add_implicit_explicit_matrix_times_f_evaluations_to(self, matrix_implicit, matrix_explicit, rhs):
        lvl = self.level
        P = lvl.prob

        buffer = P.dtype_u(P.init, val=0.0)
        for m in range(self.coll.num_nodes):
            recvBuf = buffer if m == self.rank else None
            self.comm.Reduce(
                lvl.dt
                * (
                    matrix_implicit[m + 1, self.rank + 1] * lvl.f[self.rank + 1].impl
                    + matrix_explicit[m + 1, self.rank + 1] * lvl.f[self.rank + 1].expl
                ),
                recvBuf,
                root=m,
                op=MPI.SUM,
            )
        rhs[self.rank] += buffer[:]

    def add_matrix_times_f_evaluations_to(self, matrix, rhs):
        lvl = self.level
        P = lvl.prob

        buffer = P.dtype_u(P.init, val=0.0)
        for m in range(self.coll.num_nodes):
            recvBuf = buffer if m == self.rank else None
            self.comm.Reduce(
                lvl.dt * matrix[m + 1, self.rank + 1] * (lvl.f[self.rank + 1].impl + lvl.f[self.rank + 1].expl),
                recvBuf,
                root=m,
                op=MPI.SUM,
            )
        rhs[self.rank] += buffer[:]

    def add_integral_of_right_hand_side_at_last_node_only(self, integral):
        lvl = self.level
        P = lvl.prob

        recvBuf = P.dtype_u(P.init, val=0.0) if self.rank == self.coll.num_nodes - 1 else None
        self.comm.Reduce(
            lvl.dt * self.coll.Qmat[-1, self.rank + 1] * (lvl.f[self.rank + 1].impl + lvl.f[self.rank + 1].expl),
            recvBuf,
            root=self.coll.num_nodes - 1,
            op=MPI.SUM,
        )

        if self.rank == self.coll.num_nodes - 1:
            integral[self.rank] += recvBuf

    def extrapolate_to_end_point(self):
        lvl = self.level

        self.comm.Allreduce(
            lvl.dt * self.coll.weights[self.rank] * (lvl.f[self.rank + 1].impl + lvl.f[self.rank + 1].expl),
            lvl.uend,
            op=MPI.SUM,
        )
        lvl.uend += lvl.u[0]

        # add up tau correction of the full interval (last entry)
        if lvl.tau[-1] is not None:
            lvl.uend += lvl.tau[-1]

    def add_new_information_from_forward_substitution(self, rhs, current_node):
        pass
