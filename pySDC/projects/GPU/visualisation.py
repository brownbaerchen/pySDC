from mpi4py import MPI
import matplotlib.pyplot as plt
import os
import gc
import numpy as np
import pickle
from pySDC.projects.GPU.utils import PathFormatter
from pySDC.helpers.stats_helper import get_sorted


def plot_solution_Brusselator(axs, procs, idx):
    plotting_args = {'vmin': 0, 'vmax': 7}
    for n in range(procs[2]):
        solution = V.get_solution([procs[0], procs[1], n], idx)
        x, y = V.get_grid([procs[0], procs[1], n])
        for i in [0, 1]:
            axs[i].pcolormesh(x, y, solution['u'][i], **plotting_args)


def combine_stats():
    stats = {}

    for p3 in range(procs[2]):
        path_args = {**kwargs, 'num_procs': [procs[0], procs[1], p3 + 1]}
        with open(
            PathFormatter.complete_fname(
                name='stats', format='pickle', base_path=f'{V.log_solution.path}', **path_args
            ),
            'rb',
        ) as file:
            stats = {**stats, **pickle.load(file)}
    return stats


def plot_solution_Schroedinger(ax, procs, idx):
    plotting_args = {'vmin': 0, 'vmax': 1}
    for n in range(procs[2]):
        solution = V.get_solution([procs[0], procs[1], n], idx)
        x, y = V.get_grid([procs[0], procs[1], n])
        ax.pcolormesh(x, y, abs(solution['u']), **plotting_args)


def plot_solution_AC(ax, procs, idx):
    plotting_args = {'vmin': 0, 'vmax': 1, 'rasterized': True}
    for n in range(procs[2]):
        solution = V.get_solution([procs[0], procs[1] - 1, n], idx)
        x, y = V.get_grid([procs[0], procs[1] - 1, n])
        ax.pcolormesh(x, y, solution['u'], **plotting_args)
    ax.set_aspect(1.0)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(f'$t$ = {solution["t"]:.2f}')


def plot_solution_GS(ax, procs, idx):
    plotting_args = {'vmin': 0, 'vmax': 0.5, 'rasterized': True}
    for n in range(procs[2]):
        solution = V.get_solution([procs[0], procs[1] - 1, n], idx)
        x, y = V.get_grid([procs[0], procs[1] - 1, n])
        ax.pcolormesh(x, y, solution['u'][1], **plotting_args)
        del x
        del y
    ax.set_aspect(1.0)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(f'$t$ = {solution["t"]:.2f}')
    del solution


def plot_restarts(stats, ax, relative=True):
    restart = get_sorted(stats, recomputed=None, type='restart')
    restart_ = [me[1] for me in restart]
    if relative:
        ax.plot([me[0] for me in restart], np.cumsum(restart_) / np.cumsum(np.ones_like(restart_)))
        ax.set_ylabel(r'relative number of restarts')
    else:
        ax.plot([me[0] for me in restart], np.cumsum(restart_))
        ax.set_ylabel(r'number of restarts')
    ax.set_xlabel(r'$t$')


def plot_step_size(stats, ax):
    # with open(f'{V.log_solution.path}/{type(V.prob).__name__}_stats.pickle', 'rb') as file:
    #     data = pickle.load(file)

    dt = get_sorted(stats, recomputed=False, type='dt')
    ax.plot([me[0] for me in dt], [me[1] for me in dt])
    ax.set_ylabel(r'$\Delta t$')
    ax.set_xlabel(r'$t$')


def plot_grid(ax, procs):
    plotting_args = {'color': 'white'}
    for n in range(procs[2]):
        x, y = V.get_grid([procs[0], procs[1] - 1, n])

        ax.axvline(x.min(), **plotting_args)
        ax.axhline(y.min(), **plotting_args)


def plot_Brusselator():
    fig, axs = plt.subplots(1, 2)
    procs = [0, 0, 4]
    plot_solution_Brusselator(axs, procs, 0)
    plot_grid(axs[0], procs)
    plot_grid(axs[1], procs)


def plot_Schroedinger():
    fig, ax = plt.subplots()
    procs = [0, 0, 4]
    plot_solution_Schroedinger(ax, procs, 0)
    plot_grid(ax, procs)


def plot_AC(procs):
    fig, ax = plt.subplots()
    plot_solution_AC(ax, procs, 0)
    plot_grid(ax, procs)


def plot_all(func=None, format='png', redo=False):
    func = func if func else problem.plot

    for i in range(0, 999999, comm.size):
        path = PathFormatter.complete_fname(
            base_path='./simulation_plots',
            name='solution',
            problem=V.prob,
            num_procs=procs,
            space_resolution=space_resolution,
            restart_idx=i + comm.rank,
            space_levels=space_levels,
            format=format,
        )

        if os.path.isfile(path) and not redo:
            continue

        try:
            fig = func(V, procs, i + comm.rank)
            fig.savefig(path, bbox_inches='tight', dpi=300)
            print(f'Stored {path!r}', flush=True)

            for ax in fig.get_axes():
                del ax
            fig.clf()
            plt.close(fig)
            del fig
        except FileNotFoundError:
            break

        gc.collect()


def format_procs(_procs):
    return f'{_procs[0]-1}_{_procs[1]-1}_{_procs[2]-1}'


def make_video(name, format='png'):
    import subprocess

    if comm.rank > 0:
        return None

    fname = PathFormatter.complete_fname(
        name='solution',
        problem=V.prob,
        num_procs=procs,
        space_resolution=space_resolution,
        space_levels=space_levels,
    )
    path = f'./simulation_plots/{fname}_%06d.{format}'

    cmd = f'ffmpeg -y -i {path} -pix_fmt yuv420p -r 9 -s 2048:1536 videos/{fname}_{name}.mp4'
    print(f'Recorded video to "videos/{fname}_{name}.mp4"')
    subprocess.run(cmd.split())


def plot_time_dep_AC_thing(ax):
    t = np.linspace(0, problem.default_Tend, 1000)
    desc = problem(comm_world=MPI.COMM_SELF).get_default_description()
    params = desc['problem_params']
    ax.plot(
        t,
        desc['problem_class'].get_time_dep_fac(params['time_freq'], params['time_dep_strength'], t),
        color='black',
        label='Time dependent modulation',
    )


if __name__ == '__main__':
    from configs import get_experiment, parse_args

    kwargs = parse_args()
    problem = kwargs['problem']
    procs = kwargs['procs']
    space_resolution = kwargs['space_resolution']
    space_levels = kwargs['space_levels']

    V = get_experiment('visualisation')(**kwargs)
    PathFormatter.log_solution = V.log_solution
    comm = MPI.COMM_WORLD

    plot_all(redo=True)
    comm.Barrier()

    if comm.rank == 0:
        make_video('AC_adaptivity')
        stats = combine_stats()

        fig, axs = plt.subplots(3, 1, sharex=True)

        plot_time_dep_AC_thing(axs[1])
        plot_step_size(stats, axs[0])
        plot_restarts(stats, axs[2], relative=False)
        for ax in axs[:-1]:
            ax.set_xlabel('')
        fig.savefig(f'plots/step_size_{type(V.prob).__name__}_{format_procs(procs)}.pdf')

        if comm.size == 1:
            plt.show()
