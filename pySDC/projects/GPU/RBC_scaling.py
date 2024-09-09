import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.projects.GPU.configs import get_config
from pySDC.projects.GPU.run_experiment import parse_args
from pySDC.helpers.stats_helper import get_sorted


def plot_scaling_space_parallel(args, list_procs_space):

    timings = {}

    for procs_space in list_procs_space:
        procs = [1, 1, procs_space]

        config = get_config(args['config'], procs)

        path = f'data/{config.get_path(ranks=[me -1 for me in procs])}-stats.pickle'
        with open(path, 'rb') as file:
            stats = pickle.load(file)

        timing_step = get_sorted(stats, type='timing_step')

        timings[procs_space] = np.mean([me[1] for me in timing_step])

    fig, ax = plt.subplots()
    ax.plot(timings.keys(), timings.values())
    ax.loglog(timings.keys(), timings[1] / np.array(list(timings.keys())), ls='--', color='grey')
    fig.savefig('plots/RBC_space_scaling')


if __name__ == '__main__':
    args = parse_args()
    plot_scaling_space_parallel(args, list_procs_space=[1, 2, 4])
