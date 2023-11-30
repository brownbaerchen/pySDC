def plot_interpolation_error(**kwargs):  # pragma: no cover
    import numpy as np
    import matplotlib.pyplot as plt
    from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal
    from pySDC.core.Lagrange import LagrangeApproximation
    from pySDC.core.Collocation import CollBase

    setup_mpl()
    from pySDC.projects.Resilience.collocation_adaptivity import CMAP

    def getStep(dt, num_nodes, quad_type, degree):
        """
        Get a step prepared for polynomial test equation

        Args:
            dt (float): Step size
            num_nodes (int): Number of nodes
            quad_type (str): Type of quadrature

        Returns:
           (dict): Stats object generated during the run
           (pySDC.Controller.controller): Controller used in the run
        """
        from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
        from pySDC.implementations.convergence_controller_classes.estimate_polynomial_error import (
            EstimatePolynomialError,
        )

        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        # initialize level parameters
        level_params = {}
        level_params['dt'] = dt
        level_params['restol'] = 1.0

        # initialize sweeper parameters
        sweeper_params = {}
        sweeper_params['quad_type'] = quad_type
        sweeper_params['num_nodes'] = num_nodes

        problem_params = {'degree': degree}

        # initialize step parameters
        step_params = {}
        step_params['maxiter'] = 0

        # initialize controller parameters
        controller_params = {}
        controller_params['logger_level'] = 30
        controller_params['mssdc_jac'] = False

        # fill description dictionary for easy step instantiation
        description = {}
        description['problem_class'] = polynomial_testequation
        description['problem_params'] = problem_params
        description['sweeper_class'] = sweeper_class
        description['sweeper_params'] = sweeper_params
        description['level_params'] = level_params
        description['step_params'] = step_params
        description['convergence_controllers'] = {EstimatePolynomialError: {}}

        return controller_nonMPI(num_procs=1, controller_params=controller_params, description=description).MS[0]

    # prepare figure
    fig, ax = plt.subplots(figsize=figsize_by_journal('JSC_beamer', 0.5, 0.75))

    args = {
        'num_nodes': 3,
        'quad_type': 'RADAU-RIGHT',
        'dt': 1.0,
        'degree': 6,
        **kwargs,
    }

    # prepare variables
    step = getStep(**args)
    level = step.levels[0]
    prob = level.prob
    nodes = np.append([0], level.sweep.coll.nodes)

    # initialize variables
    step.status.slot = 0
    step.status.iter = 1
    level.status.time = 0.0
    level.status.residual = 0.0
    level.u[0] = prob.u_exact(t=0)
    level.sweep.predict()
    M = len(level.u)
    interpolate_on_node = (
        M - 2
    )  # M = coll.num_nodes + 1, so this is shifted wrt to the implementation in the convergence controller

    for i in range(len(level.u)):
        if level.u[i] is not None:
            level.u[i][:] = prob.u_exact(nodes[i] * level.dt)
    u = np.array(level.u)

    # plot exact solution
    _coll = CollBase(100, 0, args['dt'], quad_type='LOBATTO')
    t = _coll.nodes
    ax.plot(t, [prob.u_exact(me) for me in t], label=f'Degree {args["degree"]} exact solution', color=CMAP[0])

    # plot SDC interpolated polynomial
    interpolator = LagrangeApproximation(points=nodes)
    interpolation_matrix = interpolator.getInterpolationMatrix(t)
    u_SDC = interpolation_matrix @ u
    ax.plot(t, u_SDC, label=rf'Degree {level.sweep.coll.order} SDC solution', color=CMAP[1], ls='--')
    ax.scatter(nodes, u, color=CMAP[1])

    # plot secondary interpolation
    reduced_points = np.arange(M) != interpolate_on_node
    interpolator = LagrangeApproximation(points=nodes[reduced_points])
    interpolation_matrix = interpolator.getInterpolationMatrix(t)
    u_reduced = interpolation_matrix @ u[reduced_points]
    ax.plot(t, u_reduced, label=rf'Degree {len(nodes[reduced_points])-1} approximation', color=CMAP[2], ls='-.')
    ax.scatter(nodes[reduced_points], u[reduced_points], color=CMAP[2])

    # plot error
    u_reduced_at_node = u_reduced[np.argmin((t - nodes[interpolate_on_node]) ** 2)]
    ax.plot(
        [nodes[interpolate_on_node]] * 2,
        [u[interpolate_on_node], u_reduced_at_node],
        color='black',
        marker='x',
        label=f'Order {len(nodes[reduced_points])} error estimate',
        ls=':',
    )

    ax.legend(frameon=False)
    ax.set_xlabel(r'$t$')
    fig.tight_layout()
    fig.savefig('data/paper/polynomial_interpolation_error.pdf')


if __name__ == "__main__":
    plot_interpolation_error(num_nodes=3, degree=6)
