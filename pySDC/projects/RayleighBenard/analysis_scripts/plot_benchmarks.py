import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pySDC.projects.RayleighBenard.analysis_scripts.plotting_utils import figsize, savefig, get_plotting_style

COLORS = {'JUSUF': 'tab:blue', 'BOOSTER': 'tab:orange', 'JUPITER': 'tab:green'}
RA_TO_RES = {
    '1e5': r'$N=128\times 128\times 32$',
    '1e6': r'$N=256\times 256\times 64$',
    '1e7': r'$N=512\times 512\times 128$',
    '1e8': r'$N=1024\times 1024\times 256$',
}


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
    savefig(fig, 'pySDC_GPU_timings')


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


def plot_binding():
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.4))
    all_data = [
        pd.read_csv(f'benchmarks/results/{machine}_RBC3DG4R4SDC44Ra{Ra}.txt')
        for machine, Ra in zip(['JUSUF', 'JUPITER'], ['1e5', '1e7'])
    ]

    for ax, data in zip(axs.flatten(), all_data):
        binds = ['block:cyclic:cyclic', 'cyclic:cyclic:cyclic']
        dists = ['space_first', 'time_first']
        for dist in dists:
            dist_label = 'space-major' if dist[0] == 's' else 'time-major'
            ls = '-' if dist[0] == 's' else '--'
            for bind in binds:
                bind_label = 'block' if bind[0] == 'b' else 'cyclic'
                ms = '.' if bind[0] == 'b' else 'x'
                color = 'tab:blue' if bind[0] == 'b' else 'tab:orange'
                mask = np.logical_and(data.distribution == bind, data.binding == dist)
                mask = np.logical_and(mask, data.ntasks_time > 1)
                ax.loglog(
                    data.procs[mask],
                    data.time[mask],
                    label=f'{dist_label}, {bind_label}',
                    ls=ls,
                    marker=ms,
                    color=color,
                )
        ax.legend(frameon=False)
        XPU = 'GPU' if any(np.isfinite(data.time_GPU)) else 'CPU'
        _res = data.res[0]
        ax.set_title(fr'$N={{{_res*4}}}\times{{{_res*4}}} \times {{{_res}}}$ {XPU}')
        ax.set_xlabel(r'$N_\mathrm{tasks}$')
        ax.set_ylabel('time / s')
    fig.tight_layout()
    savefig(fig, f'pySDC_binding')


def plot_scaling(Ra):
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.4), sharex=True)

    machines = ['JUSUF', 'BOOSTER', 'JUPITER']

    for machine in machines:
        try:
            data = pd.read_csv(f'benchmarks/results/{machine}_RBC3DG4R4SDC44Ra{Ra}.txt')
        except FileNotFoundError:
            continue
        bind_mask = data.distribution == 'block:cyclic:cyclic'
        dist_mask = data.binding == 'time_first'
        base_mask = np.logical_and(bind_mask, dist_mask)

        mask = base_mask

        finite_mask = np.logical_and(base_mask, np.isfinite(data.time))
        ref_idx = np.argmin(data.procs[finite_mask])
        ref_time = np.array(data.time[finite_mask])[ref_idx]
        ref_procs = np.array(data.procs[finite_mask])[ref_idx]
        ref_procs_time = np.array(data.ntasks_time[finite_mask])[ref_idx]
        print(
            f'Reference time for {Ra=} on {machine} is {ref_time}s with {ref_procs} procs and {ref_procs_time} procs in time'
        )
        for _tasks_time in np.unique(data.ntasks_time):
            ls = '-' if _tasks_time == 1 else '--'
            PinT_label = '' if _tasks_time == 1 else ' PinT'

            mask = np.logical_and(data.ntasks_time == _tasks_time, base_mask)
            axs[0].loglog(
                data.procs[mask], data.time[mask], color=COLORS[machine], ls=ls, label=f'{machine}{PinT_label}'
            )

            speedup = ref_time / data.time
            efficiency = speedup / (data.procs / ref_procs)
            axs[1].loglog(
                data.procs[mask], efficiency[mask], color=COLORS[machine], ls=ls, label=f'{machine}{PinT_label}'
            )

    axs[1].set_yscale('linear')
    axs[0].set_xlabel(r'$N_\mathrm{tasks}$')
    axs[1].set_xlabel(r'$N_\mathrm{tasks}$')
    axs[0].set_ylabel(r'time / s')
    axs[1].set_ylabel(r'parallel efficiency')
    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False)
    fig.tight_layout()
    savefig(fig, f'scaling_{Ra}')


def compare_methods_single_config(ax, machine, Ra='1e5', normalize=False):
    methods = ['SDC44', 'SDC23', 'RK', 'Euler']

    if normalize:
        norm_data = pd.read_csv(f'benchmarks/results/{machine}_RBC3DG4R4EulerRa{Ra}.txt')
        norm = norm_data.time[0]
        for cost in [3, 4, 5, 13]:
            ax.axhline(cost, ls=':', color='black')
    else:
        norm_data = None
        norm = 1

    for method in methods:
        config = f"RBC3DG4R4{method}Ra{Ra}"
        try:
            data = pd.read_csv(f'benchmarks/results/{machine}_{config}.txt')
        except FileNotFoundError:
            continue
        bind_mask = data.distribution == 'block:cyclic:cyclic'
        dist_mask = data.binding == 'time_first'
        base_mask = np.logical_and(bind_mask, dist_mask)

        mask = base_mask

        finite_mask = np.logical_and(base_mask, np.isfinite(data.time))
        for _tasks_time in np.unique(data.ntasks_time):
            mask = np.logical_and(data.ntasks_time == _tasks_time, base_mask)
            plotting_style = get_plotting_style(config)
            plotting_style['ls'] = '-' if _tasks_time == 1 else '--'
            plotting_style['label'] += ' PinT' if _tasks_time > 1 else ''

            timings = np.array(data.time[mask])
            procs = np.array(data.procs[mask])
            space_procs = np.array(data.ntasks_space[mask])

            if norm_data is not None:
                for i in range(len(timings)):
                    ref_time = np.array(norm_data.time[norm_data.ntasks_space == space_procs[i]])[0]
                    timings[i] /= ref_time
                # print(f'{machine} {config} {_tasks_time} {np.min(data.time[mask]) / np.min(norm_data.time):.2f}')
            ax.loglog(procs, timings, **plotting_style)

    if normalize:
        ax.set_ylabel(r'time / ($t_\mathrm{E}+t_\mathrm{S}$)')
    else:
        ax.set_ylabel(r'time / s')

    ax.set_xlabel(r'$N_\mathrm{tasks}$')
    ax.set_title(RA_TO_RES[Ra])
    # ax.legend(frameon=False)


def compare_methods(machine, normalize=False):
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.4))

    Ras = {'JUSUF': ['1e5', '1e6'], 'BOOSTER': ['1e6', '1e7'], 'JUPITER': ['1e6', '1e7']}
    for Ra, ax in zip(Ras[machine], axs.flatten(), strict=True):
        compare_methods_single_config(ax, machine, Ra, normalize=normalize)

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # removes duplicates

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),  # centered below figure
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    if normalize:
        fig.savefig(f"plots/compare_methods_{machine}_normalized.pdf", bbox_inches="tight")
    else:
        fig.savefig(f"plots/compare_methods_{machine}.pdf", bbox_inches="tight")


def plot_space_scaling(method='Euler'):
    fig, axs = plt.subplots(1, 3, figsize=figsize(scale=1, ratio=0.4), sharex=True, sharey=True)

    for machine in ['JUSUF', 'BOOSTER', 'JUPITER']:
        for Ra, ax in zip(['1e5', '1e6', '1e7'], axs.flatten()):
            try:
                data = pd.read_csv(f'benchmarks/results/{machine}_RBC3DG4R4{method}Ra{Ra}.txt')
            except FileNotFoundError:
                continue

            mask = np.isfinite(data.time)
            time = np.array(data.time[mask])
            procs = np.array(data.procs[mask])

            time_max = time[0]
            time_min = np.min(time)
            procs_max = procs[0]
            procs_min = procs_max * time_max / time_min
            ax.loglog([procs_max, procs_min], [time_max, time_min], color='black', ls=':', label='ideal')

            ax.loglog(procs, time, color=COLORS[machine], label=f'{machine}')
            ax.set_title(RA_TO_RES[Ra])

    for ax in axs.flatten():
        ax.set_xlabel(r'$N_\mathrm{tasks}$')
        ax.set_box_aspect(1)
    axs[0].set_ylabel(r'time / s')

    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # removes duplicates

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),  # centered below figure
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    savefig(fig, f'space_scaling_{method}')


def plot_space_time_scaling(method='SDC44'):
    fig, axs = plt.subplots(1, 3, figsize=figsize(scale=1, ratio=0.4))  # , sharex=True, sharey=True)
    PinT_efficiency_fig, PinT_efficiency_ax = plt.subplots(figsize=figsize(scale=1, ratio=0.4))

    for machine in ['JUSUF', 'BOOSTER', 'JUPITER']:
        for Ra, ax in zip(['1e5', '1e6', '1e7'], axs.flatten()):
            try:
                data = pd.read_csv(f'benchmarks/results/{machine}_RBC3DG4R4{method}Ra{Ra}.txt')
            except FileNotFoundError:
                continue

            base_mask = np.logical_and(data.distribution == 'block:cyclic:cyclic', data.binding == 'time_first')
            base_mask = np.logical_and(base_mask, np.isfinite(data.time))

            for ntasks_time in np.unique(data.ntasks_time):
                mask = np.logical_and(base_mask, data.ntasks_time == ntasks_time)
                time = np.array(data.time[mask])
                procs = np.array(data.procs[mask])

                plotting_style = {}
                plotting_style['ls'] = '-' if ntasks_time == 1 else '--'
                plotting_style['label'] = machine + ' PinT' if ntasks_time > 1 else ''
                plotting_style['color'] = COLORS[machine]

                time_max = time[0]
                time_min = np.min(time)
                procs_max = procs[0]
                procs_min = procs_max * time_max / time_min
                ax.loglog([procs_max, procs_min], [time_max, time_min], color='black', ls=':', label='ideal')

                ax.loglog(procs, time, **plotting_style)
                ax.set_title(RA_TO_RES[Ra])

                if ntasks_time > 1:  # plot PinT efficiency
                    ntasks_space = procs // ntasks_time
                    time_s = np.array(
                        [
                            np.nanmin(
                                np.array(
                                    data.time[np.logical_and(data.ntasks_time == 1, data.ntasks_space == _ntasks_space)]
                                )
                            )
                            for _ntasks_space in ntasks_space
                        ]
                    )
                    plotting_style = {}
                    plotting_style['color'] = COLORS[machine]
                    plotting_style['marker'] = {'1e5': '.', '1e6': 'x', '1e7': 'o'}[Ra]
                    plotting_style['ls'] = {'JUSUF': '-', 'BOOSTER': '-.', 'JUPITER': '--'}[machine]
                    plotting_style['label'] = f'{RA_TO_RES[Ra]} {machine}'
                    PinT_efficiency_ax.plot(ntasks_space, time_s / time / ntasks_time, **plotting_style)
                    PinT_efficiency_ax.set_xscale('log')

    for ax in axs.flatten():
        ax.set_xlabel(r'$N_\mathrm{tasks}$')
        ax.set_box_aspect(1)
        ax.set_ylabel(r'time / s')

    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # removes duplicates

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),  # centered below figure
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    savefig(fig, f'space_time_scaling_{method}')

    handles, labels = PinT_efficiency_ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # removes duplicates

    PinT_efficiency_ax.set_ylabel('PinT Efficiency')
    PinT_efficiency_ax.set_xlabel(r'$N_\mathrm{tasks,\ space}$')
    PinT_efficiency_fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),  # centered below figure
        ncol=2,
        frameon=False,
    )
    PinT_efficiency_fig.tight_layout()
    savefig(PinT_efficiency_fig, f'PinT_efficiency_{method}')


if __name__ == '__main__':
    # for machine in ['JUSUF', 'BOOSTER', 'JUPITER']:
    #    # plot_binding(machine)
    #    compare_methods(machine, normalize=False)
    # compare_methods('JUPITER')
    # plot_binding()
    # plot_space_scaling()
    plot_space_time_scaling('SDC44')
    plt.show()
