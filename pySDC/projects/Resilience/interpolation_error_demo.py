from pySDC.core.Hooks import hooks


class LogSolutionAllNodes(hooks):
    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u',
            value=L.u.copy(),
        )


class LogErrorPostIter(hooks):
    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)
        L = step.levels[level_number]
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e',
            value=abs(L.u[-1] - L.prob.u_exact(L.time + L.dt)),
        )


def getController(dt, num_nodes, quad_type, degree, prob='poly'):
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
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.estimate_polynomial_error import (
        EstimatePolynomialError,
    )

    if prob == 'poly':
        from pySDC.implementations.problem_classes.polynomial_test_problem import (
            polynomial_testequation as problem_class,
        )
    elif prob == 'vdp':
        from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol as problem_class
    elif prob == 'test':
        from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class

    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

    # initialize level parameters
    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = -1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = quad_type
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['initial_guess'] = 'zero'

    problem_params = {}
    if prob == 'poly':
        problem_params['degree'] = degree
    elif prob == 'vdp':
        problem_params['mu'] = 0.0
    elif prob == 'test':
        problem_params['lambdas'] = [[-2.0 + 0j]]
        problem_params['u0'] = 1.0

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 10

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['mssdc_jac'] = False
    controller_params['hook_class'] = [LogSolutionAllNodes, LogErrorPostIter]

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {EstimatePolynomialError: {}}

    return controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)


def plot_embedded_error(**kwargs):  # pragma: no cover
    import numpy as np
    import matplotlib.pyplot as plt
    from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal
    from pySDC.core.Lagrange import LagrangeApproximation
    from pySDC.core.Collocation import CollBase
    from pySDC.helpers.stats_helper import get_sorted

    setup_mpl()
    from pySDC.projects.Resilience.collocation_adaptivity import CMAP

    # prepare figure
    fig, ax = plt.subplots(figsize=figsize_by_journal('JSC_beamer', 0.5, 0.75))

    ks = [1, 2, 3, 4, 5, 6]
    dts = [1e-1, 5e-2, 1e-2, 8e-3, 1e-3]
    errors = {k: [] for k in ks}

    def single_run(dt, ks, dts, errors):
        args = {
            'num_nodes': 3,
            'quad_type': 'RADAU-RIGHT',
            'dt': dt,
            'degree': 6,
            'prob': 'test',
            **kwargs,
        }

        # prepare variables
        controller = getController(**args)
        step = controller.MS[0]
        level = step.levels[0]
        prob = level.prob
        nodes = np.append([0], level.sweep.coll.nodes) * args['dt']

        u_0 = prob.u_exact(t=0)
        _, stats = controller.run(u0=u_0, t0=0, Tend=args['dt'])

        e = get_sorted(stats, sortby='iter', type='e')
        for k in ks:
            errors[k] += [e[k - 1][1]]

    for dt in dts:
        single_run(dt, ks, dts, errors)

    for k in ks:
        ax.loglog(dts, errors[k], label=rf'$k$={k}')

    # # plot error
    # u_reduced_at_node = u_reduced[np.argmin((t - nodes[interpolate_on_node]) ** 2)]
    # ax.plot(
    #     [nodes[interpolate_on_node]] * 2,
    #     [u[interpolate_on_node], u_reduced_at_node],
    #     color='black',
    #     marker='x',
    #     label=f'Order {len(nodes[reduced_points])} error estimate',
    #     ls=':',
    # )

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'local error')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig('data/paper/embedded_error.pdf')
    plt.show()


def plot_interpolation_error(**kwargs):  # pragma: no cover
    import numpy as np
    import matplotlib.pyplot as plt
    from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal
    from pySDC.core.Lagrange import LagrangeApproximation
    from pySDC.core.Collocation import CollBase

    setup_mpl()
    from pySDC.projects.Resilience.collocation_adaptivity import CMAP

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
    step = getController(**args).MS[0]
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
    ax.plot(t, [prob.u_exact(me) for me in t], label=f'Degree {args["degree"]} polynomial', color=CMAP[0])

    # plot SDC interpolated polynomial
    interpolator = LagrangeApproximation(points=nodes)
    interpolation_matrix = interpolator.getInterpolationMatrix(t)
    u_SDC = interpolation_matrix @ u
    ax.plot(t, u_SDC, label=rf'Degree {len(nodes)-1} SDC approximation', color=CMAP[1], ls='--')
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
    plot_embedded_error()
    # plot_interpolation_error(num_nodes=3, degree=6)
