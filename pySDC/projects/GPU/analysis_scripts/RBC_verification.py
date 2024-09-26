import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.projects.GPU.configs import get_config
from pySDC.projects.GPU.run_experiment import parse_args
from pySDC.helpers.stats_helper import get_sorted


def plot_Nusselt(args):
    config = get_config(args['config'], args['procs'])
    print(config)

    path = f'data/{config.get_path(ranks=[me -1 for me in args["procs"]])}-stats.pickle'
    print(path)
    with open(path, 'rb') as file:
        stats = pickle.load(file)

    Nusselt = get_sorted(stats, type='Nusselt', recomputed=False)

    fig, ax = plt.subplots()
    colors = {
        'V': 'blue',
        't': 'green',
        'b': 'magenta',
        't_no_v': 'green',
        'b_no_v': 'magenta',
    }
    for key in ['V', 't', 'b']:
        ax.plot([me[0] for me in Nusselt], [me[1][f'{key}'] for me in Nusselt], label=f'{key}', color=colors[key])
        ax.axhline(np.mean([me[1][f'{key}'] for me in Nusselt]), color=colors[key], ls='--')
    ax.legend(frameon=False)
    ax.set_ylabel('Nusselt numbers')
    ax.set_xlabel('$t$')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    plot_Nusselt(args)
