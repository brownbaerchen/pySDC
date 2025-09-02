from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class implicit_SL(generic_implicit):
    """
    Implicit sweeper for semi-Lagrangian implementations

    Attributes:
        QI: lower triangular matrix
    """

    def integrate(self, Q=None):
        """This is hacky. It does not actually compute the integral, but we need this for the residual!"""
        L = self.level
        P = L.prob
        nodes = L.dt * self.coll.nodes
        M = self.coll.num_nodes
        Q = L.dt * self.coll.Qmat if Q is None else Q

        integral = []
        for m in range(0, M):
            integral.append(P.u_init)

            departure_points = P.get_departure_points(L.u[m + 1], nodes[m])
            integral[m] += P.interpolate(L.u[0], departure_points)
            integral[m] -= L.u[0]

            for j in range(M):
                departure_points = P.get_departure_points(L.u[m + 1], nodes[m] - nodes[j])
                integral[m] += P.interpolate(Q[m + 1, j + 1] * L.f[j + 1], departure_points)
        return integral

    def integrate2(self, Q=None):
        """This is hacky. It does not actually compute the integral, but we need this for the residual!"""
        L = self.level
        P = L.prob
        nodes = L.dt * self.coll.nodes
        M = self.coll.num_nodes
        Q = L.dt * self.coll.Qmat if Q is None else Q

        integral = []

        integral.append(P.u_init)
        departure_points = P.get_departure_points(L.u[1], nodes[0])
        integral[0] += P.interpolate(L.u[0], departure_points)
        for m in range(1, M):
            integral.append(P.u_init)

            departure_points = P.get_departure_points(L.u[m + 1], nodes[m] - nodes[m - 1])
            integral[m] += P.interpolate(integral[m - 1], departure_points)

        for m in range(0, M):
            integral[m] -= L.u[0]

            for j in range(M):
                departure_points = P.get_departure_points(L.u[m + 1], nodes[m] - nodes[j])
                integral[m] += P.interpolate(Q[m + 1, j + 1] * L.f[j + 1], departure_points)
        return integral

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        L = self.level
        P = L.prob
        M = self.coll.num_nodes
        nodes = L.dt * self.coll.nodes
        Q = L.dt * self.coll.Qmat
        Qd = L.dt * self.QI

        # only if the level has been touched before
        assert L.status.unlocked

        # prepare right hand side with everything known from previous iteration
        rhs = self.integrate(Q=Q - Qd)
        for m in range(0, M):
            rhs[m] += L.u[0]

        # do the sweep
        for m in range(0, M):

            # add forward substitution part to right hand side
            for j in range(0, m):
                if Qd[m + 1, j + 1] != 0:
                    departure_points = P.get_departure_points(L.u[m + 1], nodes[m] - nodes[j])
                    rhs[m] += P.interpolate(Qd[m + 1, j + 1] * L.f[j + 1], departure_points)

            # implicit solve
            alpha = L.dt * self.QI[m + 1, m + 1]
            if alpha == 0:
                L.u[m + 1] = rhs[m]
            else:
                L.u[m + 1] = P.solve_system(rhs[m], alpha, L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            # update function evaluations
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None
