from pySDC.core.hooks import Hooks


def compute_residual(self, stage=''):
    """
    Computation of the residual using the collocation matrix Q

    Args:
        stage (str): The current stage of the step the level belongs to
    """

    # get current level and problem description
    L = self.level

    # Check if we want to skip the residual computation to gain performance
    # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
    if stage in self.params.skip_residual_computation:
        L.status.residual = 0.0 if L.status.residual is None else L.status.residual
        return None

    # check if there are new values (e.g. from a sweep)
    # assert L.status.updated

    # compute the residual for each node

    # build QF(u)
    res_norm_all = {
        'diff': [],
        'alg': [],
        'alg_fix': [],
        'fix': [],
    }
    res = self.integrate()
    res_fix = self.integrate()
    for m in range(self.coll.num_nodes):
        res[m] += L.u[0] - L.u[m + 1]
        res_fix[m][0] += L.u[0][0] - L.u[m + 1][0]
        # add tau if associated
        if L.tau[m] is not None:
            res[m] += L.tau[m]
        # use abs function from data type here
        res_norm_all['diff'].append(abs(res[m][0]))
        res_norm_all['alg'].append(abs(res[m][1]))
        res_norm_all['alg_fix'].append(abs(res[m][1] - L.u[0][1] + L.u[m + 1][1]))
        res_norm_all['fix'].append(abs(res_fix[m]))

    res_norm = res_norm_all['fix']

    # find maximal residual over the nodes
    if L.params.residual_type == 'full_abs':
        L.status.residual = max(res_norm)
    elif L.params.residual_type == 'last_abs':
        L.status.residual = res_norm[-1]
    elif L.params.residual_type == 'full_rel':
        L.status.residual = max(res_norm) / abs(L.u[0])
    elif L.params.residual_type == 'last_rel':
        L.status.residual = res_norm[-1] / abs(L.u[0])
    else:
        raise ParameterError(
            f'residual_type = {L.params.residual_type} not implemented, choose '
            f'full_abs, last_abs, full_rel or last_rel instead'
        )

    # indicate that the residual has seen the new values
    L.status.updated = False

    return {key: max(res_norm_all[key]) for key in res_norm_all.keys()}


class ResidualHook(Hooks):
    def post_iteration(self, step, level_number):
        residual = compute_residual(step.levels[level_number].sweep)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'residual',
            value=residual,
        )


def plot_residual():
    import numpy as np
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.problem_classes.HeatEquation_Chebychov import Heat1DChebychovPreconditioning
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.stats_helper import get_sorted

    dt = 1
    Tend = 1

    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1e-13

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    problem_params = {'a': -9, 'b': 3, 'poly_coeffs': (0, 0, 0, -1, 1), 'nvars': 2**5}

    step_params = {}
    step_params['maxiter'] = 100

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [ResidualHook]

    description = {}
    description['problem_class'] = Heat1DChebychovPreconditioning
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)

    import matplotlib.pyplot as plt

    residual = get_sorted(stats, type='residual', sortby='iter')

    for name in residual[0][1].keys():
        ls = '--' if name == 'fix' else '-'
        plt.plot([me[0] for me in residual], [me[1][name] for me in residual], label=name, ls=ls)
    plt.yscale('log')
    plt.ylabel('residual')
    plt.xlabel('k')
    plt.legend(frameon=False)
    plt.show()


if __name__ == '__main__':
    plot_residual()
