from pySDC.projects.GPU.configs import AdaptivityExperiment, SingleGPUExperiment, RunAllenCahn, RunSchroedinger
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal
import matplotlib.pyplot as plt
import numpy as np


class PlotExperiments:
    experiment_cls = None
    problem = None
    num_nodes_parallel_gpu = []
    num_nodes_serial_gpu = []
    num_nodes_parallel_cpu = []
    num_nodes_serial_cpu = []

    def __init__(self, **kwargs):
        self.experiment = self.experiment_cls(problem=self.problem, **kwargs)

    def get_multiple_data(self, vary_keys, prob_args=None):
        times = {}
        num_items = [len(vary_keys[key]) for key in vary_keys.keys()]
        prob_args = {**self.experiment.prob_args, **prob_args} if prob_args else self.experiment.prob_args

        for i in range(min(num_items)):
            kwargs = {**prob_args, **{key: vary_keys[key][i] for key in vary_keys.keys()}}
            prob = self.problem(**kwargs)
            data = prob.get_data(self.experiment.name)
            fac = 4 if kwargs['useGPU'] else 48
            times[np.prod(vary_keys['num_procs'][i]) / fac] = data['times']
        return times

    def get_vary_keys(self, parallel_sweeper, useGPU):
        vary_keys = {}
        if parallel_sweeper and useGPU:
            vary_keys['num_procs'] = [[1, 4, me] for me in self.num_nodes_parallel_gpu]
        elif not parallel_sweeper and useGPU:
            vary_keys['num_procs'] = [[1, 1, 4 * me] for me in self.num_nodes_serial_gpu]
        elif parallel_sweeper and not useGPU:
            vary_keys['num_procs'] = [[1, 4, me * 12] for me in self.num_nodes_parallel_cpu]
        elif not parallel_sweeper and not useGPU:
            vary_keys['num_procs'] = [[1, 1, me * 48] for me in self.num_nodes_serial_cpu]
        else:
            raise NotImplementedError
        return vary_keys

    def plot(self, ax):
        for useGPU in [True, False]:
            for parallel_sweeper in [True, False]:
                self.plot_single(ax, parallel_sweeper, useGPU)

    def plot_single(self, ax, parallel_sweeper, useGPU):
        ls = {
            True: '-',
            False: '--',
        }
        marker = {
            False: '^',
            True: 'x',
        }
        label = {
            True: 'space-time-parallel',
            False: 'space-parallel',
        }
        label_GPU = {
            True: 'GPU',
            False: 'CPU',
        }

        timings = self.get_multiple_data(self.get_vary_keys(parallel_sweeper, useGPU), prob_args={'useGPU': useGPU})
        ax.loglog(
            timings.keys(),
            timings.values(),
            label=f'{label[parallel_sweeper]} {label_GPU[useGPU]}',
            marker=marker[useGPU],
            ls=ls[parallel_sweeper],
            markersize=7,
        )
        ax.legend(frameon=True)
        ax.set_xlabel('Nodes on JUWELS Booster')
        ax.set_ylabel('wall time / s')


class PlotSingleGPUStrongScalingSchroedinger(PlotExperiments):
    experiment_cls = SingleGPUExperiment
    problem = RunSchroedinger
    num_nodes_parallel_gpu = [1, 2, 4, 8, 16, 32, 64, 128]
    num_nodes_serial_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_parallel_cpu = [4, 8, 16]
    num_nodes_serial_cpu = [1, 4, 8, 16]


class PlotAdaptivityStrongScalingSchroedinger(PlotExperiments):
    experiment_cls = AdaptivityExperiment
    problem = RunSchroedinger
    num_nodes_parallel_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_serial_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_parallel_cpu = []
    num_nodes_serial_cpu = []


if __name__ == '__main__':
    setup_mpl()

    figsize = figsize_by_journal('Springer_Numerical_Algorithms', 0.7, 0.8)
    fig, ax = plt.subplots(figsize=figsize)
    # plotter = PlotSingleGPUStrongScalingSchroedinger()
    plotter = PlotAdaptivityStrongScalingSchroedinger()
    plotter.plot(ax)
    fig.tight_layout()
    fig.savefig('/Users/thomasbaumann/Desktop/space_time_SDC_Schroedinger.pdf', bbox_inches='tight')

    plt.show()
