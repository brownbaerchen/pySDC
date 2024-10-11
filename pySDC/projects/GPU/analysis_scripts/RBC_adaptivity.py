import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.projects.GPU.run_experiment import parse_args
from pySDC.helpers.stats_helper import get_sorted, get_list_of_types

from pySDC.projects.GPU.configs.RBC_configs import (
    RayleighBenard_dt_k_adaptivity_increment,
    RayleighBenard_dt_adaptivity,
)

Tend = 15


def plot_cost(stats, axs, **plotting_params):
    work_factorizations = get_sorted(stats, type='work_factorizations')
    axs[0].plot(
        [me[0] for me in work_factorizations], np.cumsum([me[1] for me in work_factorizations]), **plotting_params
    )
    axs[0].set_yscale('log')
    axs[0].set_ylabel('number of factorizations')

    work_rhs = get_sorted(stats, type='work_rhs')
    axs[1].plot([me[0] for me in work_rhs], np.cumsum([me[1] for me in work_rhs]), **plotting_params)
    axs[1].set_ylabel('number of RHS evaluations')
    axs[1].set_xlabel('$t$')


def load_stats(args):

    config = get_config(args)
    procs = args['procs']

    path = f'data/{config.get_path(ranks=[me -1 for me in procs])}-stats-whole-run.pickle'
    with open(path, 'rb') as file:
        stats = pickle.load(file)
    return stats


def check_solutions(args, configs):
    idx = 84

    t = []
    u = []
    for conf in configs:
        args['config'] = conf
        config = get_config(args)

        logToFile = config.get_LogToFile()
        data = logToFile.load(idx)
        u += [data['u']]
        t += [data['t']]

    for i in range(1, len(configs)):
        print(
            f'Diff in time: {abs(t[i] - t[0]):.6e}, diff in solution: {np.linalg.norm((u[i] - u[0]).flatten(), np.inf):.2e}'
        )


if __name__ == '__main__':
    args = parse_args()
    fig, axs = plt.subplots(2, 1, sharex=True)

    # check_solutions(args, ['RBC_dt_00', 'RBC_dt_01', 'RBC_dt_02', 'RBC_dt_03'])

    for min_slope in [0, 1, 2, 3, 10]:
        if min_slope < 10:
            args['config'] = f'RBC_dt_0{min_slope}'
        else:
            args['config'] = f'RBC_dt_{min_slope}'
        stats = load_stats(args)
        plot_cost(stats, axs, label=f'{min_slope/10}')
    axs[0].legend()
    plt.show()
