from pySDC.helpers.stats_helper import get_sorted


def get_description(QI='MIN-SR-S', maxiter=5, gamma=1):
    if QI == 'ARK':
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3 as sweeper_class

        maxiter = 1
    else:
        from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class
    from pySDC.implementations.problem_classes.HeatEquation_Chebychev import Heat1DUltrasphericalTimeDepBCs

    level_params = {}
    level_params['dt'] = 1e-3
    level_params['restol'] = 1e-9

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 2
    sweeper_params['QI'] = QI
    sweeper_params['QE'] = 'PIC'
    sweeper_params['initial_guess'] = 'copy'

    problem_params = {'nu': 1.0e1, 'gamma': gamma}

    step_params = {}
    step_params['maxiter'] = maxiter

    description = {}
    description['problem_class'] = Heat1DUltrasphericalTimeDepBCs
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description


def record_order(dts, desc, Tend):
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    errors = []

    controller_params = {}
    controller_params['logger_level'] = 90
    controller_params['hook_class'] = [LogGlobalErrorPostRun]
    controller_params['mssdc_jac'] = False

    for dt in dts:
        desc['level_params']['dt'] = dt

        controller_args = {
            'controller_params': controller_params,
            'description': desc,
        }
        controller = controller_nonMPI(**controller_args, num_procs=1)
        P = controller.MS[0].levels[0].prob

        t0 = 0.0
        uinit = P.u_exact(t0)

        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        error = get_sorted(stats, type='e_global_post_run')
        errors.append(error[-1][1])
    return errors


def get_order(dts, errors):
    return [np.log(errors[i + 1] / errors[i]) / np.log(dts[i + 1] / dts[i]) for i in range(len(dts) - 1)]


if __name__ == '__main__':
    import numpy as np

    for QI in ['LU', 'ARK']:
        desc = get_description(QI=QI, maxiter=99, gamma=1e1)
        Tend = 1e0
        dts = [1 / 2**me for me in range(4, 10)]
        errors = record_order(dts, desc, Tend)

        order = get_order(dts, errors)
        print(order)
        print(f'Order: {np.mean(order):.2f}')

        import matplotlib.pyplot as plt

        plt.loglog(dts, errors)
        plt.loglog(dts, [dt**2 for dt in dts])
    plt.show()
