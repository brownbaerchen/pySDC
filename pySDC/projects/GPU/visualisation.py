from configs import Visualisation, RunBrusselator, RunSchroedinger, RunAllenCahn, RunAllenCahnForcing, RunGS
import matplotlib.pyplot as plt
import os


problem = RunGS
V = Visualisation(problem=problem, useGPU=False)


# for i in range(99):
#     for n_space in range(4):
#         V.plot(ax, [0, 0, n_space], i)
#         plt.pause(1e-7)
def plot_solution_Brusselator(axs, ranks, idx):
    plotting_args = {'vmin': 0, 'vmax': 7}
    for n in range(ranks[2]):
        solution = V.get_solution([ranks[0], ranks[1], n], idx)
        x, y = V.get_grid([ranks[0], ranks[1], n])
        for i in [0, 1]:
            axs[i].pcolormesh(x, y, solution['u'][i], **plotting_args)


def plot_solution_Schroedinger(ax, ranks, idx):
    plotting_args = {'vmin': 0, 'vmax': 1}
    for n in range(ranks[2]):
        solution = V.get_solution([ranks[0], ranks[1], n], idx)
        x, y = V.get_grid([ranks[0], ranks[1], n])
        ax.pcolormesh(x, y, abs(solution['u']), **plotting_args)


def plot_solution_AC(ax, ranks, idx):
    plotting_args = {'vmin': 0, 'vmax': 1, 'rasterized': True}
    for n in range(ranks[2]):
        solution = V.get_solution([ranks[0], ranks[1] - 1, n], idx)
        x, y = V.get_grid([ranks[0], ranks[1] - 1, n])
        ax.pcolormesh(x, y, solution['u'], **plotting_args)
    ax.set_aspect(1.0)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(f'$t$ = {solution["t"]:.2f}')


def plot_solution_GS(ax, ranks, idx):
    plotting_args = {'vmin': 0, 'vmax': 0.5, 'rasterized': True}
    for n in range(ranks[2]):
        solution = V.get_solution([ranks[0], ranks[1] - 1, n], idx)
        x, y = V.get_grid([ranks[0], ranks[1] - 1, n])
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

    with open(f'{V.logger_hook.path}/{type(V.prob).__name__}_stats.pickle', 'rb') as file:
        data = pickle.load(file)
    dt = data['dt']
    ax.plot([me[0] for me in dt], [me[1] for me in dt])
    ax.set_ylabel(r'$\Delta t$')
    ax.set_xlabel(r'$t$')


def plot_grid(ax, ranks):
    plotting_args = {'color': 'white'}
    for n in range(ranks[2]):
        x, y = V.get_grid([ranks[0], ranks[1] - 1, n])

        ax.axvline(x.min(), **plotting_args)
        ax.axhline(y.min(), **plotting_args)


def plot_Brusselator():
    fig, axs = plt.subplots(1, 2)
    ranks = [0, 0, 4]
    plot_solution_Brusselator(axs, ranks, 0)
    plot_grid(axs[0], ranks)
    plot_grid(axs[1], ranks)


def plot_Schroedinger():
    fig, ax = plt.subplots()
    ranks = [0, 0, 4]
    plot_solution_Schroedinger(ax, ranks, 0)
    plot_grid(ax, ranks)


def plot_AC(ranks):
    fig, ax = plt.subplots()
    plot_solution_AC(ax, ranks, 0)
    plot_grid(ax, ranks)


def plot_all(ranks, func, format='png', redo=False):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    for i in range(0, 999999, comm.size):
        fname = f'solution_{type(V.prob).__name__}_{ranks[0]}_{ranks[1]-1}_{ranks[2]-1}'
        path = f'./simulation_plots/{fname}_{V.logger_hook.format_index(i+comm.rank)}.{format}'

        if os.path.isfile(path) and not redo:
            continue

        try:
            fig, ax = plt.subplots()
            func(ax, ranks, i + comm.rank)
            fig.savefig(path, bbox_inches='tight', dpi=300)
            print(f'Stored {path!r}', flush=True)
            ax.cla()
            fig.clf()
            plt.close(fig)
        except FileNotFoundError:
            break


fig, ax = plt.subplots()
plot_step_size(ax)
fig.savefig('out/step_size.pdf')
# plot_solution_GS(ax, [0,1,4], 0)
plot_all([0, 1, 1], plot_solution_GS, redo=True)
# plot_AC(ranks)
# plot_all(ranks, plot_AC)
plt.show()
