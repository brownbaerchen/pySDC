import numpy as np
import pickle
from pySDC.projects.GPU.configs import RayleighBenardRegular
from pySDC.projects.GPU.run_experiment import run_experiment, parse_args

from pySDC.core.convergence_controller import ConvergenceController


class ReachTendExactly(ConvergenceController):

    def setup(self, controller, params, description, **kwargs):
        defaults = {
            "control_order": +50,
            "Tend": None,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def get_new_step_size(self, controller, step, **kwargs):
        L = step.levels[0]
        L.status.dt_new = min([L.params.dt, self.params.Tend - L.time - L.dt])


class RayleighBenardOrder(RayleighBenardRegular):
    experiment_name = 'NumOrder'
    t_inc = 1e-2
    dt = 1e-1

    def get_description(self, *args, dt=1e-2, order='3', **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['level_params']['dt'] = dt
        desc['level_params']['restol'] = -1

        if order == 3:
            desc['sweeper_params']['num_nodes'] = 2
            desc['step_params']['maxiter'] = 3
        elif order == 4:
            desc['sweeper_params']['num_nodes'] = 3
            desc['step_params']['maxiter'] = 4
        elif order == 5:
            desc['sweeper_params']['num_nodes'] = 3
            desc['step_params']['maxiter'] = 5

        desc['convergence_controllers'].pop(CFLLimit)
        desc['convergence_controllers'][ReachTendExactly] = {'Tend': self.Tend}

        # desc['problem_params']['Rayleigh'] = 2e3
        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        regular = RayleighBenardRegular(n_procs_list=self.n_procs_list, comm_world=self.comm_world)
        u, t = regular.get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)
        type(self).Tend = t + self.t_inc

        return u, t

    def get_LogToFile(self, ranks=None):
        return []


def get_order(starting_idx, increment, order):

    args = parse_args()
    args['restart_idx'] = starting_idx

    config = RayleighBenardOrder(n_procs_list=args['procs'])
    desc = config.get_description()
    P = desc['problem_class'](**desc['problem_params'])
    _, t = config.get_initial_condition(P, restart_idx=starting_idx)

    config.t_inc = increment

    dts = increment / 2 ** np.arange(6)
    dts = np.array([increment / me for me in [1, 2, 4, 5, 10, 20]])

    u = {}

    path = f'data/numerical_order-{config.comm_world.rank}-{starting_idx}-{order}.pickle'
    for dt in dts:
        u[dt] = run_experiment(args, config, dt=dt, order=order)

    with open(path, 'wb') as file:
        pickle.dump({'t': t, 'u': u, 'increment': increment}, file)


def compute_orders(starting_idx, increment):
    for order in [3, 4, 5]:
        get_order(starting_idx, increment, order)


def plot_orders(starting_idx):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS

    cmap = list(TABLEAU_COLORS)

    fig, ax = plt.subplots()

    orders = [3, 4, 5]

    args = parse_args()
    args['restart_idx'] = starting_idx

    config = RayleighBenardOrder(n_procs_list=args['procs'])
    # desc = config.get_description()
    # P = desc['problem_class'](**desc['problem_params'])
    # _, t = config.get_initial_condition(P, restart_idx=starting_idx)
    # if config.comm_world.rank > 0:
    #     return None

    for order in orders:
        path = f'data/numerical_order-{config.comm_world.rank}-{starting_idx}-{order}.pickle'
        with open(path, 'rb') as file:
            data = pickle.load(file)
        u = data['u']

        errors = []
        idx = 0
        dts = np.array(list(u.keys()))
        for i in range(len(dts) - 1):
            errors += [abs(u[dts[i]][idx] - u[dts[i + 1]][idx])]

        errors = np.array(errors)

        num_order = np.log(errors[1:] / errors[:-1]) / np.log(dts[1:-1] / dts[:-2])

        ax.loglog(dts[:-1], errors, color=cmap[order - min(orders)], label=f'Order {order}')
        ax.loglog(dts[:-1], errors[0] * (dts[:-1] / dts[0]) ** order, color=cmap[order - min(orders)], ls='--')

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylabel('global error')
    ax.set_xlabel(r'$\Delta t$')
    ax.legend(frameon=False)
    ax.set_title(f't={data["t"]:.2f}')
    fig.savefig(f'plots/RBC-order-{starting_idx}.pdf')
    plt.show()


if __name__ == '__main__':
    # get_order(70, run=False, increment=1)
    # get_order(130, increment=5e-2, order=3)
    idx = 130
    compute_orders(130, increment=5e-2)
    plot_orders(130)
