import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pySDC.projects.RayleighBenard.analysis_scripts.plotting_utils import figsize, savefig


def plot_config(fname, ax, key='time'):  # pragma: no cover
    data = pd.read_csv(fname)

    for _res in np.unique(data.res):
        res_mask = data.res == _res

        for _tasks_time in np.unique(data.ntasks_time):
            task_mask = data.ntasks_time == _tasks_time
            Pint_label = " PinT" if _tasks_time > 1 else ""

            for _distribution in np.unique(data.distribution):
                distribution_mask = data.distribution == _distribution

                mask = np.logical_and(np.logical_and(res_mask, task_mask), distribution_mask)
                procs = np.array(data.procs[mask])
                time = np.array(getattr(data, key)[mask])
                ax.loglog(
                    procs[np.argsort(procs)],
                    time[np.argsort(procs)],
                    label=fr'$N_z$={_res}{Pint_label} {_distribution}',
                )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('tasks')
    ax.set_ylabel('time / s')


def plot_CPU_timings():  # pragma: no cover
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.4), sharex=True)

    data_Ra1e5 = pd.read_csv('benchmarks/results/JUSUF_RBC3DG4R4SDC44Ra1e5.txt')
    data_Ra1e6 = pd.read_csv('benchmarks/results/JUSUF_RBC3DG4R4SDC44Ra1e6.txt')

    ref = data_Ra1e5.time[0]
    mask_cyclic = data_Ra1e5.distribution == 'cyclic:cyclic:cyclic'
    mask_time_serial = data_Ra1e5.ntasks_time == 1
    mask = np.logical_and(mask_cyclic, mask_time_serial)
    axs[0].loglog(data_Ra1e5.procs[mask], data_Ra1e5.time[mask], label=r'$128^2\times 32$')
    axs[1].loglog(
        data_Ra1e5.procs[mask], ref / data_Ra1e5.time[mask] / data_Ra1e5.procs[mask], label=r'$128^2\times 32$'
    )

    mask = np.logical_and(mask_cyclic, ~mask_time_serial)
    axs[0].loglog(data_Ra1e5.procs[mask], data_Ra1e5.time[mask], label=r'$128^2\times 32$ PinT')
    axs[1].loglog(
        data_Ra1e5.procs[mask], ref / data_Ra1e5.time[mask] / data_Ra1e5.procs[mask], label=r'$128^2\times 32$ PinT'
    )

    ref = data_Ra1e6.time[0] * data_Ra1e6.procs[0]
    mask_cyclic = data_Ra1e6.distribution == 'cyclic:cyclic:cyclic'
    mask_time_serial = data_Ra1e6.ntasks_time == 1
    mask = np.logical_and(mask_cyclic, mask_time_serial)
    axs[0].loglog(data_Ra1e6.procs[mask], data_Ra1e6.time[mask], label=r'$256^2\times 64$')
    axs[1].loglog(
        data_Ra1e6.procs[mask], ref / data_Ra1e6.time[mask] / data_Ra1e6.procs[mask], label=r'$256^2\times 64$'
    )

    mask = np.logical_and(mask_cyclic, ~mask_time_serial)
    axs[0].loglog(data_Ra1e6.procs[mask], data_Ra1e6.time[mask], label=r'$256^2\times 64$ PinT')
    axs[1].loglog(
        data_Ra1e6.procs[mask], ref / data_Ra1e6.time[mask] / data_Ra1e6.procs[mask], label=r'$256^2\times 64$ PinT'
    )

    axs[1].set_yscale('linear')
    axs[0].set_xlabel(r'$N_\mathrm{tasks}$')
    axs[1].set_xlabel(r'$N_\mathrm{tasks}$')
    axs[0].set_ylabel(r'time / s')
    axs[1].set_ylabel(r'parallel efficiency')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    savefig(fig, 'CPU_timings.pdf')


def plot_GPU_timings():  # pragma: no cover
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.4), sharex=True)

    data_Ra1e5 = pd.read_csv('benchmarks/results/BOOSTER_RBC3DG4R4SDC44Ra1e5.txt')
    data_Ra1e6 = pd.read_csv('benchmarks/results/BOOSTER_RBC3DG4R4SDC44Ra1e6.txt')

    ref = data_Ra1e5.time[0]
    mask_cyclic = data_Ra1e5.distribution == 'cyclic:cyclic:cyclic'
    mask_time_serial = data_Ra1e5.ntasks_time == 1
    mask = np.logical_and(mask_cyclic, mask_time_serial)
    axs[0].loglog(data_Ra1e5.procs[mask], data_Ra1e5.time_GPU[mask], label=r'$128^2\times 32$')
    axs[1].loglog(
        data_Ra1e5.procs[mask], ref / data_Ra1e5.time_GPU[mask] / data_Ra1e5.procs[mask], label=r'$128^2\times 32$'
    )

    mask = np.logical_and(mask_cyclic, ~mask_time_serial)
    axs[0].loglog(data_Ra1e5.procs[mask], data_Ra1e5.time_GPU[mask], label=r'$128^2\times 32$ PinT')
    axs[1].loglog(
        data_Ra1e5.procs[mask], ref / data_Ra1e5.time_GPU[mask] / data_Ra1e5.procs[mask], label=r'$128^2\times 32$ PinT'
    )

    ref = data_Ra1e6.time[5] * data_Ra1e6.procs[5]
    mask_cyclic = data_Ra1e6.distribution == 'cyclic:cyclic:cyclic'
    mask_time_serial = data_Ra1e6.ntasks_time == 1
    mask = np.logical_and(mask_cyclic, mask_time_serial)
    axs[0].loglog(data_Ra1e6.procs[mask], data_Ra1e6.time[mask], label=r'$256^2\times 64$')
    axs[1].loglog(
        data_Ra1e6.procs[mask], ref / data_Ra1e6.time[mask] / data_Ra1e6.procs[mask], label=r'$256^2\times 64$'
    )

    mask = np.logical_and(mask_cyclic, ~mask_time_serial)
    axs[0].loglog(data_Ra1e6.procs[mask], data_Ra1e6.time[mask], label=r'$256^2\times 64$ PinT')
    axs[1].loglog(
        data_Ra1e6.procs[mask], ref / data_Ra1e6.time[mask] / data_Ra1e6.procs[mask], label=r'$256^2\times 64$ PinT'
    )

    axs[1].set_yscale('linear')
    axs[0].set_xlabel(r'$N_\mathrm{tasks}$')
    axs[1].set_xlabel(r'$N_\mathrm{tasks}$')
    axs[0].set_ylabel(r'time / s')
    axs[1].set_ylabel(r'parallel efficiency')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    savefig(fig, 'GPU_timings.pdf')


if __name__ == '__main__':
    plot_CPU_timings()
    plot_GPU_timings()
    plt.show()
