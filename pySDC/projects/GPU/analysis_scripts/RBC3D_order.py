import os
import pickle
import numpy as np
from pySDC.helpers.fieldsIO import FieldsIO
from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D
from mpi4py import MPI
import matplotlib.pyplot as plt

step_sizes = {
    'RBC3DG4Ra1e5': [3, 1e0, 1e-1, 8e-2, 6e-2],
    'RBC3DG4RKRa1e5': [1e3, 5, 4, 1e0, 1e-1, 8e-2, 6e-2],
    'RBC3DG4R4Ra1e5': [1e0, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2],
    'RBC3DG4R4RKRa1e5': [1e0, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2],
}
n_freefall_times = {}


def no_logging_hook(*args, **kwargs):
    return None


def get_path(args):
    config = get_config(args)
    fname = config.get_file_name()
    print(fname.index('dt'))
    return f'{fname[:fname.index('dt')]}order.pickle'


def compute_errors(args, dts, Tend):
    errors = {'u': [], 'v': [], 'w': [], 'T': [], 'p': [], 'dt': []}
    prob = RayleighBenard3D(nx=4, ny=4, nz=4)

    dts = np.sort(dts)[::-1]
    ref = run(args, dts[-1], Tend)
    for dt in dts[:-1]:
        u = run(args, dt, Tend)
        e = u - ref
        for comp in ['u', 'v', 'w', 'T', 'p']:
            i = prob.index(comp)
            e_comp = np.max(np.abs(e[i]))
            e_comp = MPI.COMM_WORLD.allreduce(e_comp, op=MPI.MAX)
            errors[comp].append(e_comp)
        errors['dt'].append(dt)
        print(errors)

    path = get_path(args)
    with open(path, 'wb') as file:
        pickle.dump(errors, file)
        print(f'Saved errors to {path}', config.get_file_name())


def plot_errors(args):
    with open(get_path(args), 'rb') as file:
        errors = pickle.load(file)

    for comp in errors.keys():
        plt.loglog(errors['dt'], errors[comp], label=comp)

    plt.loglog(errors['dt'], np.array(errors['dt']) ** 4, label='Order 4')
    plt.loglog(errors['dt'], np.array(errors['dt']) ** 2, label='Order 2')
    plt.legend()
    plt.show()


def run(args, dt, Tend):
    from pySDC.projects.GPU.run_experiment import run_experiment
    from pySDC.core.errors import ConvergenceError

    args['mode'] = 'run'
    args['dt'] = dt

    config = get_config(args)
    config.Tend = n_freefall_times.get(type(config).__name__, 3)
    ic_config_name = type(config).__name__.replace('RK', '')
    config.ic_config = type(get_config({**args, 'config': ic_config_name}))

    config.get_LogToFile = no_logging_hook
    config.Tend = Tend

    u = run_experiment(args, config)
    return u


if __name__ == '__main__':
    from pySDC.projects.GPU.run_experiment import parse_args

    args = parse_args()
    config = get_config(args)

    # run(args, 1e-3, 1e-2)
    # compute_errors(args, [8e-2, 4e-2, 2e-2, 1e-2, 5e-3], 8e-2)
    plot_errors(args)

    # dts = step_sizes[type(config).__name__]
    # stability = [None for _ in dts]
