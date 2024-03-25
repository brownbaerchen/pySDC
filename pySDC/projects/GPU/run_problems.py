import numpy as np
from mpi4py import MPI
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.helpers.stats_helper import get_sorted
import pickle
import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal

setup_mpl()


def get_comms(n_procs_list, comm_world=None, _comm=None, _tot_rank=0, _rank=None):
    comm_world = MPI.COMM_WORLD if comm_world is None else comm_world
    _comm = comm_world if _comm is None else _comm
    _rank = comm_world.rank if _rank is None else _rank

    if len(n_procs_list) > 0:
        color = _tot_rank + _rank // n_procs_list[0]
        new_comm = comm_world.Split(color)
        return [new_comm] + get_comms(
            n_procs_list[1:],
            comm_world,
            _comm=new_comm,
            _tot_rank=_tot_rank + _comm.size * new_comm.rank,
            _rank=_comm.rank // new_comm.size,
        )
    else:
        return []


class RunProblem:
    default_Tend = 0.0

    def __init__(
        self,
        custom_description=None,
        custom_controller_params=None,
        num_procs=None,
        comm_world=None,
        imex=False,
        useGPU=True,
        space_resolution=None,
    ):
        num_procs = (
            [
                1,
            ]
            * 3
            if num_procs is None
            else num_procs
        )
        assert len(num_procs) == 3

        custom_description = {} if custom_description is None else custom_description
        custom_controller_params = {} if custom_controller_params is None else custom_controller_params

        self.useGPU = useGPU
        self.num_procs = num_procs
        self.comm_steps, self.comm_sweep, self.comm_space = get_comms(num_procs, comm_world=comm_world)
        self.get_description(custom_description, imex)
        self.get_controller_params(custom_controller_params)
        if space_resolution:
            self.set_space_resolution(space_resolution)

    def get_default_description(self):
        description = {
            'step_params': {},
            'level_params': {},
            'sweeper_params': {},
            'sweeper_class': None,
            'problem_params': {},
            'problem_class': None,
            'convergence_controllers': {},
        }
        description['problem_params']['useGPU'] = self.useGPU
        return description

    def set_space_resolution(self, resolution):
        current_resolution = self.get_space_resolution()
        self.description['problem_params']['nvars'] = (int(resolution),) * len(current_resolution)

    def get_space_resolution(self):
        return self.description['problem_params']['nvars']

    @property
    def get_poly_adaptivity_default_params(self):
        defaults = {
            'restol_rel': None,
            'e_tol_rel': None,
            'restart_at_maxiter': True,
            'restol_min': 1e-12,
            'restol_max': 1e-5,
            'factor_if_not_converged': 4.0,
            'residual_max_tol': 1e9,
            'maxiter': self.description['sweeper_params'].get('maxiter', 99),
            'interpolate_between_restarts': True,
            'abort_at_growing_residual': True,
        }
        return defaults

    def add_polynomial_adaptivity(self, params=None):
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError

        params = params if params else {}
        self.description['convergence_controllers'][AdaptivityPolynomialError] = {
            **self.get_poly_adaptivity_default_params,
            **params,
        }

    def get_sweeper(self, imex):
        useMPI = self.comm_sweep.size > 1

        if imex and useMPI:
            from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI as sweeper_class
        elif imex and not useMPI:
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class
        else:
            raise NotImplementedError

        self.description['sweeper_class'] = sweeper_class
        if useMPI:
            if self.useGPU:
                from pySDC.helpers.NCCL_communicator import NCCLComm

                self.description['sweeper_params']['comm'] = NCCLComm(self.comm_sweep)
            else:
                self.description['sweeper_params']['comm'] = self.comm_sweep

    def get_description(self, custom_description, imex):
        self.description = self.get_default_description()
        self.get_sweeper(imex)
        self.description = merge_descriptions(self.description, custom_description)

    def get_default_controller_params(self):
        from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

        controller_params = {'logger_level': 30, 'mssdc_jac': False, 'hook_class': [LogGlobalErrorPostRun]}
        return controller_params

    def get_controller_params(self, custom_controller_params):
        self.controller_params = merge_descriptions(self.get_default_controller_params(), custom_controller_params)

        # prevent duplicate logging
        if self.comm_space.rank > 0 or self.comm_sweep.rank > 0:
            self.controller_params['logger_level'] = 30

    def run(self, Tend=None, storeResults=True):
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        Tend = self.default_Tend if Tend is None else Tend
        controller = controller_MPI(
            description=self.description, controller_params=self.controller_params, comm=self.comm_steps
        )
        prob = controller.S.levels[0].prob
        u_init = prob.u_exact(0)
        u_end, stats = controller.run(u0=u_init, t0=0, Tend=Tend)

        return stats

    def record_timing(self, num_runs=5, name='', Tend=None):
        errors = []
        times = []

        for _ in range(num_runs):
            stats = self.run(Tend=Tend)

            errors += [max(me[1] for me in get_sorted(stats, type='e_global_post_run'))]
            times += [max(me[1] for me in get_sorted(stats, type='timing_run'))]
            if self.comm_sweep.rank == 0 and self.comm_space.rank == 0 and self.comm_steps.rank == 0:
                print(f'Needed {times[-1]:.2e}s with error {errors[-1]:.2e}', flush=True)

        if self.comm_sweep.rank == 0 and self.comm_space.rank == 0 and self.comm_steps.rank == 0 and Tend is None:
            with open(self.get_path(name=name), 'wb') as file:
                pickle.dump({'errors': errors, 'times': times}, file)
            print(f'Stored file {self.get_path(name=name)!r}')

    def get_path(self, name=''):
        import socket

        if socket.gethostname() == 'thomas-work':
            base_path = '/Users/thomasbaumann/Documents/pySDC/pySDC/projects/GPU/out'
        else:
            base_path = '/p/project/ccstma/baumann7/pySDC/pySDC/projects/GPU/out'
        procs = f'{self.num_procs[0]}x{self.num_procs[1]}x{self.num_procs[2]}'
        prob = type(self).__name__
        gpu = 'GPU' if self.useGPU else 'CPU'
        space_resolution = f'{self.get_space_resolution()[0]:d}'
        return f'{base_path}/{prob}_{procs}_{gpu}{name}_{space_resolution}.pickle'

    def get_data(self, name=''):
        with open(self.get_path(name=name), 'rb') as file:
            data = pickle.load(file)

        results = {}
        for key, values in data.items():
            results[key] = np.median(values)
            if np.std(values) / np.median(values) > 1e-2:
                print(f'Warning! Standard deviation too large in {key!r}!')
        return results


class RunAllenCahn(RunProblem):
    default_Tend = 1e-2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, imex=True)

    def get_default_description(self):
        from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

        description = super().get_default_description()

        description['step_params']['maxiter'] = 5

        description['level_params']['dt'] = 1e-4
        description['level_params']['restol'] = 1e-8

        description['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        description['sweeper_params']['num_nodes'] = 3
        description['sweeper_params']['QI'] = 'MIN-SR-S'
        description['sweeper_params']['QE'] = 'PIC'

        description['problem_params']['nvars'] = (2**10,) * 2
        description['problem_params']['init_type'] = 'circle_rand'
        description['problem_params']['L'] = 16
        description['problem_params']['spectral'] = False
        description['problem_params']['comm'] = self.comm_space

        description['problem_class'] = allencahn_imex

        return description

    @property
    def get_poly_adaptivity_default_params(self):
        defaults = super().get_poly_adaptivity_default_params
        defaults['e_tol'] = 1e-7
        return defaults


class RunSchroedinger(RunProblem):
    default_Tend = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, imex=True)

    def get_default_description(self):
        from pySDC.implementations.problem_classes.NonlinearSchroedinger_MPIFFT import nonlinearschroedinger_imex

        description = super().get_default_description()

        description['step_params']['maxiter'] = 9

        description['level_params']['dt'] = 1e-2
        description['level_params']['restol'] = 1e-8

        description['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        description['sweeper_params']['num_nodes'] = 4
        description['sweeper_params']['QI'] = 'MIN-SR-S'
        description['sweeper_params']['QE'] = 'PIC'

        description['problem_params']['nvars'] = (2**13,) * 2
        description['problem_params']['spectral'] = False
        description['problem_params']['comm'] = self.comm_space

        description['problem_class'] = nonlinearschroedinger_imex

        return description

    @property
    def get_poly_adaptivity_default_params(self):
        defaults = super().get_poly_adaptivity_default_params
        defaults['e_tol'] = 1e-7
        return defaults


def cast_to_bool(arg):
    if arg == 'False':
        return False
    else:
        return True


def parse_args():
    import sys

    allowed_args = {
        'Nsteps': int,
        'Nsweep': int,
        'Nspace': int,
        'num_runs': int,
        'useGPU': cast_to_bool,
        'space_resolution': int,
        'Tend': float,
    }

    args = {}
    for me in sys.argv[1:]:
        for key, cast in allowed_args.items():
            if key in me:
                args[key] = cast(me[len(key) + 1 :])

    return args


class Experiment:
    name = None

    def __init__(
        self, problem, num_procs=None, useGPU=True, num_runs=5, space_resolution=None, custom_description=None
    ):
        self.problem = problem
        self.num_runs = num_runs

        self.prob_args = {
            'num_procs': num_procs if num_procs else [1, 1, 1],
            'useGPU': useGPU,
            'custom_description': custom_description if custom_description else {},
            'space_resolution': space_resolution,
        }

        self.prob = problem(**self.prob_args)

    def run(self, Tend=None):
        self.prob.record_timing(num_runs=self.num_runs, name=self.name, Tend=Tend)


class SingleGPUExperiment(Experiment):
    name = 'single_gpu'


class AdaptivityExperiment(Experiment):
    name = 'adaptivity'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prob.add_polynomial_adaptivity()


class PlotExperiments:
    experiment_cls = None
    num_nodes_parallel_gpu = []
    num_nodes_serial_gpu = []
    num_nodes_parallel_cpu = []
    num_nodes_serial_cpu = []

    def __init__(self, **kwargs):
        self.experiment = self.experiment_cls(**kwargs)

    def get_multiple_data(self, vary_keys, prob_args=None):
        times = {}
        num_items = [len(vary_keys[key]) for key in vary_keys.keys()]
        prob_args = {**self.experiment.prob_args, **prob_args} if prob_args else self.experiment.prob_args

        for i in range(min(num_items)):
            kwargs = {**prob_args, **{key: vary_keys[key][i] for key in vary_keys.keys()}}
            prob = problem(**kwargs)
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


class PlotSingleGPUStrongScaling(PlotExperiments):
    experiment_cls = SingleGPUExperiment
    num_nodes_parallel_gpu = [1, 2, 4, 8, 16, 32, 64, 128]
    num_nodes_serial_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_parallel_cpu = [4, 8, 16]
    num_nodes_serial_cpu = [1, 4, 8, 16]


class PlotAdaptivityStrongScaling(PlotExperiments):
    experiment_cls = AdaptivityExperiment
    num_nodes_parallel_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_serial_gpu = [1, 2, 4, 8, 16, 32, 64]
    num_nodes_parallel_cpu = []
    num_nodes_serial_cpu = []


if __name__ == '__main__':
    problem = RunSchroedinger
    args = parse_args()

    num_procs = [args.get(key, 1) for key in ['Nsteps', 'Nsweep', 'Nspace']]

    kwargs = {
        'num_procs': num_procs,
        'num_runs': args.get('num_runs', 5),
        'useGPU': args.get('useGPU', True),
        'space_resolution': args.get('space_resolution', 2**13),
        'problem': RunSchroedinger,
    }

    # experiment = SingleGPUExperiment(**kwargs)
    experiment = AdaptivityExperiment(**kwargs)
    # experiment.run()

    figsize = figsize_by_journal('Springer_Numerical_Algorithms', 0.7, 0.8)
    fig, ax = plt.subplots(figsize=figsize)
    plotter = PlotSingleGPUStrongScaling(**kwargs)
    # plotter = PlotAdaptivityStrongScaling(**kwargs)
    plotter.plot(ax)
    fig.tight_layout()
    fig.savefig('/Users/thomasbaumann/Desktop/space_time_SDC_Schroedinger.pdf', bbox_inches='tight')

    plt.show()
