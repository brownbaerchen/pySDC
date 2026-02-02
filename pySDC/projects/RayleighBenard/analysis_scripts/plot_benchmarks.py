import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pySDC.projects.RayleighBenard.analysis_scripts.plotting_utils import figsize, savefig


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
    savefig(fig, 'pySDC_CPU_timings')


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
    savefig(fig, '/pySDC_GPU_timings')


def plot_distribution():  # pragma: no cover
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.4))

    data_Ra1e5_CPU = pd.read_csv('benchmarks/results/JUSUF_RBC3DG4R4SDC44Ra1e5.txt')
    data_Ra1e6_CPU = pd.read_csv('benchmarks/results/JUSUF_RBC3DG4R4SDC44Ra1e6.txt')
    data_Ra1e5_GPU = pd.read_csv('benchmarks/results/BOOSTER_RBC3DG4R4SDC44Ra1e5.txt')
    data_Ra1e6_GPU = pd.read_csv('benchmarks/results/BOOSTER_RBC3DG4R4SDC44Ra1e6.txt')

    mask_cyclic = data_Ra1e5_CPU.distribution == 'cyclic:cyclic:cyclic'
    mask_block = data_Ra1e5_CPU.distribution == 'block:cyclic:cyclic'
    mask_time_parallel = data_Ra1e5_CPU.ntasks_time > 1
    mask = np.logical_and(mask_cyclic, mask_time_parallel)
    axs[0].loglog(data_Ra1e5_CPU.procs[mask], data_Ra1e5_CPU.time[mask], label=r'$128^2\times 32$ c:c:c')

    mask = np.logical_and(~mask_cyclic, mask_time_parallel)
    axs[0].loglog(data_Ra1e5_CPU.procs[mask], data_Ra1e5_CPU.time[mask], label=r'$128^2\times 32$ b:c:c', ls='--')

    mask_cyclic = data_Ra1e6_CPU.distribution == 'cyclic:cyclic:cyclic'
    mask_block = data_Ra1e6_CPU.distribution == 'block:cyclic:cyclic'
    mask_time_parallel = data_Ra1e6_CPU.ntasks_time > 1
    mask = np.logical_and(mask_cyclic, mask_time_parallel)
    axs[0].loglog(data_Ra1e6_CPU.procs[mask], data_Ra1e6_CPU.time[mask], label=r'$256^2\times 64$ c:c:c')

    mask = np.logical_and(~mask_cyclic, mask_time_parallel)
    axs[0].loglog(data_Ra1e6_CPU.procs[mask], data_Ra1e6_CPU.time[mask], label=r'$256^2\times 64$ b:c:c', ls='--')

    mask_cyclic = data_Ra1e5_GPU.distribution == 'cyclic:cyclic:cyclic'
    mask_block = data_Ra1e5_GPU.distribution == 'block:cyclic:cyclic'
    mask_time_parallel = data_Ra1e5_GPU.ntasks_time > 1
    mask = np.logical_and(mask_cyclic, mask_time_parallel)
    axs[1].loglog(data_Ra1e5_GPU.procs[mask], data_Ra1e5_GPU.time[mask], label=r'$128^2\times 32$ c:c:c')

    mask = np.logical_and(~mask_cyclic, mask_time_parallel)
    axs[1].loglog(data_Ra1e5_GPU.procs[mask], data_Ra1e5_GPU.time[mask], label=r'$128^2\times 32$ b:c:c', ls='--')

    mask_cyclic = data_Ra1e6_GPU.distribution == 'cyclic:cyclic:cyclic'
    mask_block = data_Ra1e6_GPU.distribution == 'block:cyclic:cyclic'
    mask_time_parallel = data_Ra1e6_GPU.ntasks_time > 1
    mask = np.logical_and(mask_cyclic, mask_time_parallel)
    axs[1].loglog(data_Ra1e6_GPU.procs[mask], data_Ra1e6_GPU.time[mask], label=r'$256^2\times 64$ c:c:c')

    mask = np.logical_and(~mask_cyclic, mask_time_parallel)
    axs[1].loglog(data_Ra1e6_GPU.procs[mask], data_Ra1e6_GPU.time[mask], label=r'$256^2\times 64$ b:c:c', ls='--')

    axs[0].set_xlabel(r'$N_\mathrm{tasks}$')
    axs[1].set_xlabel(r'$N_\mathrm{tasks}$')
    axs[0].set_ylabel(r'time / s')
    axs[1].set_ylabel(r'time / s')
    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False)
    axs[0].set_title('JUSUF')
    axs[1].set_title('JUWELS booster')
    fig.tight_layout()
    savefig(fig, 'pySDC_space_time_distribution')


if __name__ == '__main__':
    # plot_CPU_timings()
    # plot_GPU_timings()
    plot_distribution()
    plt.show()
