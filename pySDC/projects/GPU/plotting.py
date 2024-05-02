from pySDC.projects.GPU.configs import (
    AdaptivityExperiment,
    SingleGPUExperiment,
    PFASST,
    RunAllenCahn,
    RunSchroedinger,
    RunAllenCahnForcing,
)
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
    num_procs = []
    space_resolution = {}

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
        if len(self.num_procs) > 0:
            vary_keys['num_procs'] = self.num_procs
            vary_keys['space_resolution'] = [
                self.space_resolution.get(sum(me), self.space_resolution[4]) for me in self.num_procs
            ]
        elif parallel_sweeper and useGPU:
            vary_keys['num_procs'] = [[1, 4, me] for me in self.num_nodes_parallel_gpu]
            vary_keys['space_resolution'] = [
                self.space_resolution.get(me, self.space_resolution[1]) for me in self.num_nodes_parallel_gpu
            ]
        elif not parallel_sweeper and useGPU:
            vary_keys['num_procs'] = [[1, 1, 4 * me] for me in self.num_nodes_serial_gpu]
            vary_keys['space_resolution'] = [
                self.space_resolution.get(me, self.space_resolution[1]) for me in self.num_nodes_serial_gpu
            ]
        elif parallel_sweeper and not useGPU:
            vary_keys['num_procs'] = [[1, 4, me * 12] for me in self.num_nodes_parallel_cpu]
            vary_keys['space_resolution'] = [
                self.space_resolution.get(me, self.space_resolution[1]) for me in self.num_nodes_parallel_cpu
            ]
        elif not parallel_sweeper and not useGPU:
            vary_keys['num_procs'] = [[1, 1, me * 48] for me in self.num_nodes_serial_cpu]
            vary_keys['space_resolution'] = [
                self.space_resolution.get(me, self.space_resolution[1]) for me in self.num_nodes_serial_cpu
            ]
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

        if len(timings.keys()) == 0:
            return None

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
    space_resolution = {1: 1024}


class PlotAdaptivityStrongScalingSchroedinger(PlotExperiments):
    experiment_cls = AdaptivityExperiment
    problem = RunSchroedinger
    num_nodes_parallel_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_serial_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_parallel_cpu = []
    num_nodes_serial_cpu = []
    space_resolution = {1: 1024}


class PlotAdaptivityStrongScalingACF(PlotExperiments):
    experiment_cls = AdaptivityExperiment
    problem = RunAllenCahnForcing
    num_nodes_parallel_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_serial_gpu = [1, 2, 4, 8, 16, 32]
    num_nodes_parallel_cpu = []
    num_nodes_serial_cpu = []
    space_resolution = {1: 8192}


class PlotAdaptivityWeakScalingACF(PlotExperiments):
    experiment_cls = AdaptivityExperiment
    problem = RunAllenCahnForcing
    num_nodes_parallel_gpu = [1, 4, 16, 64]
    num_nodes_serial_gpu = [1, 4, 16, 64]
    space_resolution = {1: 8192, 4: 16384, 16: 32768, 64: 65536}


class PlotPFASSTSchroedinger(PlotExperiments):
    experiment_cls = PFASST
    problem = RunSchroedinger
    num_procs = [
        [1, 1, 4],
    ]
    space_resolution = {4: 256}


def ACF_plots():
    figsize = figsize_by_journal('JSC_thesis', 1, 0.5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    plotterS = PlotAdaptivityStrongScalingACF()
    plotterW = PlotAdaptivityWeakScalingACF()
    plotterS.plot(axs[0])
    plotterW.plot(axs[1])
    for ax, label in zip(axs, ['strong scaling', 'weak scaling']):
        ax.set_box_aspect(1.0)
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig('plots/space_time_SDC_ACF.pdf')


if __name__ == '__main__':
    setup_mpl()
    plotter = PlotPFASSTSchroedinger()
    fig, ax = plt.subplots()
    plotter.plot_single(ax, True, True)

    plt.show()
