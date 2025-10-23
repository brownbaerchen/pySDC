import os
import numpy as np
from pySDC.helpers.fieldsIO import FieldsIO
from pySDC.projects.GPU.configs.base_config import get_config

step_sizes = {'RBC3DG4Ra1e5': [1e3, 1e0, 1e-1, 8e-2, 6e-2], 'RBC3DG4R4Ra1e5': [1e0, 1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2]}
n_freefall_times = {}


def get_stability(args, dt):
    args['dt'] = dt
    config = get_config(args)
    path = config.get_file_name()

    if not os.path.isfile(path):
        compute_stability(args, dt)

    file = FieldsIO.fromFile(path)
    t, u = file.readField(-1)

    reachedTend = t >= n_freefall_times.get(type(config).__name__, 3)
    isfinite = u.max() <= 1e9
    if not reachedTend:
        print(f'Did not reach Tend with {dt=:.2e}')
    elif not isfinite:
        print(f'Solution not finite with {dt=:.2e}')
    return reachedTend and isfinite


def compute_stability(args, dt):
    from pySDC.projects.GPU.run_experiment import run_experiment
    from pySDC.core.errors import ConvergenceError

    args['mode'] = 'run'
    args['dt'] = dt

    config = get_config(args)
    config.Tend = n_freefall_times.get(type(config).__name__, 3)
    config.ic_config = type(config)

    try:
        run_experiment(args, config)
    except ConvergenceError:
        pass


if __name__ == '__main__':
    from pySDC.projects.GPU.run_experiment import parse_args

    args = parse_args()
    config = get_config(args)

    dts = step_sizes[type(config).__name__]
    stability = [None for _ in dts]

    for i in range(len(dts)):
        stability[i] = get_stability(args, dts[i])
    print([(dts[i], stability[i]) for i in range(len(dts))])
