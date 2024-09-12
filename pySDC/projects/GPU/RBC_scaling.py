import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.projects.GPU.configs import get_config
from pySDC.projects.GPU.run_experiment import parse_args
from pySDC.helpers.stats_helper import get_sorted


def plot_scaling(args, procs_time, list_procs_space, ax, plot_ideal=False):

    timings = {}

    for procs_space in list_procs_space:
        procs = [1, procs_time, procs_space]

        args['procs'] = procs
        config = get_config(args)

        path = f'data/{config.get_path(ranks=[me -1 for me in procs])}-stats.pickle'
        with open(path, 'rb') as file:
            stats = pickle.load(file)

        timing_step = get_sorted(stats, type='timing_step')

        timings[procs_space * procs[1]] = np.mean([me[1] for me in timing_step])

    GPU_label = 'GPU' if args['useGPU'] else 'CPU'
    ax.plot(timings.keys(), timings.values(), label=f'{procs[1]} {GPU_label} in time', marker='x')
    if plot_ideal:
        ax.loglog(timings.keys(), timings[1] / np.array(list(timings.keys())), ls='--', color='grey')
    ax.set_xlabel(r'$N_\mathrm{procs}$')
    ax.set_ylabel(r'$t_\mathrm{step}$')


if __name__ == '__main__':
    fig, ax = plt.subplots()
    args = parse_args()
    args['config'] = 'RBC_Tibo'
    plot_scaling(args, procs_time=1, list_procs_space=[1, 2, 4, 8, 16, 32, 64], ax=ax, plot_ideal=True)
    plot_scaling(args, procs_time=4, list_procs_space=[1, 2, 4, 8, 16, 32, 64], ax=ax)
    plot_scaling({**args, 'useGPU': True}, procs_time=1, list_procs_space=[1], ax=ax)
    plot_scaling({**args, 'useGPU': True}, procs_time=4, list_procs_space=[1], ax=ax)
    ax.legend(frameon=False)
    fig.savefig('plots/RBC_space_scaling.pdf')
