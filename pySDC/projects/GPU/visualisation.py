from mpi4py import MPI
import matplotlib.pyplot as plt
import os
import gc
import numpy as np


# for i in range(99):
#     for n_space in range(4):
#         V.plot(ax, [0, 0, n_space], i)
#         plt.pause(1e-7)
def plot_solution_Brusselator(axs, procs, idx):
    plotting_args = {'vmin': 0, 'vmax': 7}
    for n in range(procs[2]):
        solution = V.get_solution([procs[0], procs[1], n], idx)
        x, y = V.get_grid([procs[0], procs[1], n])
        for i in [0, 1]:
            axs[i].pcolormesh(x, y, solution['u'][i], **plotting_args)


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


def plot_step_size(ax):
    import pickle

    # print(f'{V.logger_hook.path}/{type(V.prob).__name__}_stats.pickle', flush=True)
    with open(f'{V.logger_hook.path}/{type(V.prob).__name__}_stats.pickle', 'rb') as file:
        data = pickle.load(file)
    dt = data['dt']
    ax.plot([me[0] for me in dt], [me[1] for me in dt])
    ax.plot([me[0] for me in data['restart']], np.cumsum([me[1] for me in data['restart']])/np.cumsum(np.ones_like([me[1] for me in data['restart']])))
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
        fname = f'solution_{type(V.prob).__name__}_{procs[0]-1}_{procs[1]-1}_{procs[2]-1}'
        path = f'./simulation_plots/{fname}_{V.logger_hook.format_index(i+comm.rank)}.{format}'

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

    fname = f'solution_{type(V.prob).__name__}_{format_procs(procs)}'
    path = f'./simulation_plots/{fname}_%06d.{format}'

    cmd = f'ffmpeg -y -i {path} -pix_fmt yuv420p -r 9 -s 2048:1536 videos/{fname}_{name}.mp4'
    subprocess.run(cmd.split())


def plot_time_dep_AC_thing(ax):
    import numpy as np
    t = np.linspace(0, problem.default_Tend, 1000)
    desc = problem(comm_world = MPI.COMM_SELF).get_default_description()
    params = desc['problem_params']
    ax2 = ax.twinx()
    ax2.plot(t, desc['problem_class'].get_time_dep_fac(params['time_freq'], params['time_dep_strength'], t), color='black', label='Time dependent modulation')


if __name__ == '__main__':
    from configs import get_experiment, parse_args
    kwargs = parse_args()
    problem = kwargs['problem']
    procs = kwargs['procs']

    V = get_experiment('visualisation')(problem=problem, useGPU=False)
    comm = MPI.COMM_WORLD
    
    # plot_all(redo=False)

    if comm.rank == 0:
        # make_video('AC_adaptivity')

        fig, ax = plt.subplots()
        plot_time_dep_AC_thing(ax)
        plot_step_size(ax)
        fig.savefig(f'plots/step_size_{type(V.prob).__name__}_{format_procs(procs)}.pdf')

        if comm.size == 1:
            plt.show()
