from mpi4py import MPI
from pySDC.implementations.sweeper_classes.generic_implicit_MPI import SweeperMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order


class imex_1st_order_MPI(SweeperMPI, imex_1st_order):
    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl)

        Returns:
            list of dtype_u: containing the integral as values
        """

        L = self.level
        P = L.prob

        me = P.dtype_u(P.init, val=0.0)
        for m in range(self.coll.num_nodes):
            recvBuf = me if m == self.rank else None
            self.comm.Reduce(
                L.dt * self.coll.Qmat[m + 1, self.rank + 1] * (L.f[self.rank + 1].impl + L.f[self.rank + 1].expl),
                recvBuf,
                root=m,
                op=MPI.SUM,
            )

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        rhs = self.integrate()

        rhs -= L.dt * (
            self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1].impl
            + self.QE[self.rank + 1, self.rank + 1] * L.f[self.rank + 1].expl
        )

        # add initial value
        rhs += L.u[0]
        # add tau if associated
        if L.tau[self.rank] is not None:
            rhs += L.tau[self.rank]

        # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)

        # implicit solve with prefactor stemming from the diagonal of Qd
        L.u[self.rank + 1] = P.solve_system(
            rhs,
            L.dt * self.QI[self.rank + 1, self.rank + 1],
            L.u[self.rank + 1],
            L.time + L.dt * self.coll.nodes[self.rank],
        )
        # update function values
        L.f[self.rank + 1] = P.eval_f(L.u[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        L = self.level
        P = L.prob
        L.uend = P.dtype_u(P.init, val=0.0)

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            super().compute_end_point()
        else:
            L.uend = P.dtype_u(L.u[0])
            self.comm.Allreduce(
                L.dt * self.coll.weights[self.rank] * (L.f[self.rank + 1].impl + L.f[self.rank + 1].expl),
                L.uend,
                op=MPI.SUM,
            )
            L.uend += L.u[0]

            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]
        return None
