from configs import Visualisation, RunBrusselator, RunSchroedinger, RunAllenCahn, RunAllenCahnForcing
import matplotlib.pyplot as plt


V = Visualisation(problem=RunAllenCahnForcing, useGPU=False)


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


def plot_step_size(ax):
    import pickle

    with open(f'{V.logger_hook.path}/{type(V.prob).__name__}_stats.pickle', 'rb') as file:
        data = pickle.load(file)
    dt = data['dt']
    ax.plot([me[0] for me in dt], [me[1] for me in dt])


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


def plot_all_AC(ranks, format='png'):
    fig, ax = plt.subplots()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    for i in range(0, 999999, comm.size):
        try:
            plot_solution_AC(ax, ranks, i + comm.rank)
            path = f'./simulation_plots/{V.logger_hook.file_name}_{V.logger_hook.format_index(i+comm.rank)}.{format}'
            fig.savefig(path, bbox_inches='tight')
            ax.cla()
        except FileNotFoundError:
            break


fig, ax = plt.subplots()
ranks = [0, 1, 4]
plot_step_size(ax)
# plot_AC(ranks)
# plot_all_AC(ranks)
plt.show()
