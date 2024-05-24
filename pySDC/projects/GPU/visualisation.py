from mpi4py import MPI
import matplotlib.pyplot as plt
import os
import gc
import numpy as np
import pickle
from pySDC.projects.GPU.utils import PathFormatter
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.GPU.configs import get_experiment, parse_args, get_problem


# def plot_solution_Brusselator(axs, procs, idx):
#     plotting_args = {'vmin': 0, 'vmax': 7}
#     for n in range(procs[2]):
#         solution = V.get_solution([procs[0], procs[1], n], idx)
#         x, y = V.get_grid([procs[0], procs[1], n])
#         for i in [0, 1]:
#             axs[i].pcolormesh(x, y, solution['u'][i], **plotting_args)
#
#
# def plot_solution_Schroedinger(ax, procs, idx):
#     plotting_args = {'vmin': 0, 'vmax': 1}
#     for n in range(procs[2]):
#         solution = V.get_solution([procs[0], procs[1], n], idx)
#         x, y = V.get_grid([procs[0], procs[1], n])
#         ax.pcolormesh(x, y, abs(solution['u']), **plotting_args)
#
#
# def plot_solution_GS(ax, procs, idx):
#     plotting_args = {'vmin': 0, 'vmax': 0.5, 'rasterized': True}
#     for n in range(procs[2]):
#         solution = V.get_solution([procs[0], procs[1] - 1, n], idx)
#         x, y = V.get_grid([procs[0], procs[1] - 1, n])
#         ax.pcolormesh(x, y, solution['u'][1], **plotting_args)
#         del x
#         del y
#     ax.set_aspect(1.0)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$y$')
#     ax.set_title(f'$t$ = {solution["t"]:.2f}')
#     del solution
#
#
# def plot_grid(ax, procs):
#     plotting_args = {'color': 'white'}
#     for n in range(procs[2]):
#         x, y = V.get_grid([procs[0], procs[1] - 1, n])
#
#         ax.axvline(x.min(), **plotting_args)
#         ax.axhline(y.min(), **plotting_args)
#
#
# def plot_Brusselator():
#     fig, axs = plt.subplots(1, 2)
#     procs = [0, 0, 4]
#     plot_solution_Brusselator(axs, procs, 0)
#     plot_grid(axs[0], procs)
#     plot_grid(axs[1], procs)
#
#
# def plot_Schroedinger():
#     fig, ax = plt.subplots()
#     procs = [0, 0, 4]
#     plot_solution_Schroedinger(ax, procs, 0)
#     plot_grid(ax, procs)
#
#
# def plot_AC(procs):
#     fig, ax = plt.subplots()
#     plot_solution_AC(ax, procs, 0)
#     plot_grid(ax, procs)


class PlottingUtils:
    def __init__(self, kwargs):
        self.kwargs = kwargs.copy()
        if type(self.kwargs["problem"]) == str:
            self.kwargs["problem"] = get_problem(self.kwargs["problem"])
        self.experiment = get_experiment('visualisation')(**self.kwargs)
        PathFormatter.log_solution = self.experiment.log_solution
        self.comm = MPI.COMM_WORLD

    def combine_stats(self, kwargs):
        kwargs = {**self.kwargs, **kwargs}
        procs = kwargs['procs']

        stats = {}

        for p3 in range(procs[2]):
            path_args = {**kwargs, 'num_procs': [procs[0], procs[1], p3 + 1]}
            with open(
                PathFormatter.complete_fname(
                    name='stats', format='pickle', base_path=f'{self.experiment.log_solution.path}', **path_args
                ),
                'rb',
            ) as file:
                stats = {**stats, **pickle.load(file)}
        return stats

    @staticmethod
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

    @staticmethod
    def plot_step_size(stats, ax):
        dt = get_sorted(stats, recomputed=False, type='dt')
        ax.plot([me[0] for me in dt], [me[1] for me in dt])
        ax.set_ylabel(r'$\Delta t$')
        ax.set_xlabel(r'$t$')

    @staticmethod
    def plot_AC_radius(stats, ax):
        r = get_sorted(stats, recomputed=False, type='computed_radius')
        rE = get_sorted(stats, recomputed=False, type='exact_radius')
        ax.plot([me[0] for me in r], [me[1] for me in r], label='$r$')
        ax.plot([me[0] for me in rE], [me[1] for me in rE], label='$r^*$', ls='--')

        # compute error as maximum difference between exact and computed radius
        e = [abs(r[i][1] - rE[i][1]) for i in range(len(r))]
        print(max(e))
        # ax.plot([me[0] for me in r], e, label='e')
        ax.legend(frameon=False)
        ax.set_xlabel(r'$t$')

    def plot_all(self, func=None, format='png', redo=False):
        func = func if func else self.experiment.problem.plot
        procs = self.kwargs['procs']

        for i in range(0, 999999, self.comm.size):
            path = PathFormatter.complete_fname(
                base_path='./simulation_plots',
                name='solution',
                **{
                    **self.kwargs,
                    'restart_idx': i + self.comm.rank,
                    'format': format,
                },
            )

            if os.path.isfile(path) and not redo:
                continue

            try:
                fig = func(self.experiment, procs, i + self.comm.rank)
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
        self.comm.Barrier()

    def make_video(self, format='png'):
        if self.comm.rank > 0:
            return None
        import subprocess

        if self.comm.rank > 0:
            return None

        fname = PathFormatter.complete_fname(
            name='solution',
            **{**self.kwargs, 'restart_idx': None},
        )
        path = f'./simulation_plots/{fname}_%06d.{format}'

        cmd = f'ffmpeg -y -i {path} -pix_fmt yuv420p -r 9 -s 2048:1536 videos/{fname}.mp4'
        print(f'Recorded video to "videos/{fname}.mp4"')
        subprocess.run(cmd.split())

    def plot_time_dep_AC_thing(self, ax):
        t = np.linspace(0, self.experiment.problem.default_Tend, 1000)
        desc = self.experiment.problem(comm_world=MPI.COMM_SELF).get_default_description()
        params = desc['problem_params']
        ax.plot(
            t,
            desc['problem_class'].get_time_dep_fac(params['time_freq'], params['time_dep_strength'], t),
            color='black',
            label='Time dependent modulation',
        )


if __name__ == '__main__':
    kwargs = parse_args()

    plotter = PlottingUtils(kwargs)
    # V = get_experiment('visualisation')(**kwargs)

    plotter.plot_all(redo=True)
    plotter.make_video()

    if plotter.comm.rank == 0:
        stats = plotter.combine_stats(kwargs)

        fig, axs = plt.subplots(3, 1, sharex=True)

        # plot_time_dep_AC_thing(axs[1])
        plotter.plot_step_size(stats, axs[0])
        plotter.plot_restarts(stats, axs[2], relative=False)
        # plotter.plot_AC_radius(stats, axs[1])
        for ax in axs[:-1]:
            ax.set_xlabel('')

        fig.savefig(PathFormatter.complete_fname(base_path='plots', name='step_size', format='pdf', **kwargs))

        if plotter.comm.size == 1:
            plt.show()
