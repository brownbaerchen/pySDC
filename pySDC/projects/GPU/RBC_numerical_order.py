import pickle
from pySDC.projects.GPU.configs import RayleighBenardRegular
from pySDC.projects.GPU.run_experiment import run_experiment, parse_args


class RayleighBenardOrder(RayleighBenardRegular):
    experiment_name = 'NumOrder'
    t_inc = 1e-2
    dt = 1e-1

    def get_description(self, *args, dt=1e-2, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['level_params']['dt'] = dt
        desc['convergence_controllers'].pop(CFLLimit)
        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        regular = RayleighBenardRegular(n_procs_list=self.n_procs_list, comm_world=self.comm_world)
        u, t = regular.get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)
        type(self).Tend = t + self.t_inc

        return u, t

    def get_LogToFile(self, ranks=None):
        return []


def get_order(starting_idx, run):
    args = parse_args()
    args['restart_idx'] = starting_idx

    config = RayleighBenardOrder(n_procs_list=args['procs'])

    dts = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

    u = {}

    path = f'data/numerical_order-{config.comm_world.rank}-{starting_idx}.pickle'
    if run:
        for dt in dts:
            u[dt] = run_experiment(args, config, dt=dt)

        with open(path, 'wb') as file:
            pickle.dump(u, file)

    else:
        with open(path, 'rb') as file:
            u = pickle.load(file)

    errors = {}
    idx = 0
    for i in range(len(dts) - 1):
        errors[dts[i]] = abs(u[dts[i]][idx] - u[dts[-1]][idx])

    print(errors)


if __name__ == '__main__':
    get_order(200, run=True)
