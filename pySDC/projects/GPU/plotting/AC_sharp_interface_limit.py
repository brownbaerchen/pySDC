from pySDC.projects.GPU.utils import PathFormatter
from pySDC.projects.GPU.visualisation import PlottingUtils
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl


def plot_radii():
    kwargs = {
        'procs': [1, 4, 1],
        'useGPU': True,
        'space_levels': 1,
        'restart_idx': 200,
        'problem': 'ACI',
    }

    plotter = PlottingUtils(kwargs)
    fig, ax = plt.subplots(figsize=figsize_by_journal(journal='JSC_thesis', scale=0.8, ratio=0.6))

    restart_idx = {
        32: 98,
        64: 174,
    }
    procs = {}

    for res in [32, 64, 128, 256, 512, 1024, 2048]:
        _kwargs = kwargs.copy()
        if res in restart_idx.keys():
            _kwargs['restart_idx'] = restart_idx[res]
        if res in procs.keys():
            _kwargs['procs'] = procs[res]

        stats = plotter.combine_stats({**_kwargs, 'space_resolution': res})
        radius = get_sorted(stats, recomputed=False, type='computed_radius')

        ax.plot([me[0] for me in radius], [me[1] for me in radius], label=fr'$N={{{res}}}^2$')

    radius_exact = get_sorted(stats, recomputed=False, type='exact_radius')
    ax.plot([me[0] for me in radius_exact], [me[1] for me in radius_exact], label='exact', ls='--', color='black')
    ax.legend(frameon=False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$R$')

    fig.savefig('./plots/AC_sharp_inferface_limit.pdf', bbox_inches='tight')


if __name__ == '__main__':
    setup_mpl()
    plot_radii()
