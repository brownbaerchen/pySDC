from pySDC.projects.GPU.utils import PathFormatter
from pySDC.projects.GPU.run_problems import RunProblem
from pySDC.projects.GPU.run_problems import Experiment
from pySDC.implementations.hooks.log_solution import LogToFileAfterXs
from pySDC.projects.GPU.hooks.LogGrid import LogGrid


def parse_args():
    import argparse

    cast_to_bool = lambda me: False if me == 'False' else True

    def str_to_procs(me):
        procs = me.split('/')
        assert len(procs) == 3
        return [int(p) for p in procs]

    parser = argparse.ArgumentParser()
    parser.add_argument('--Nsteps', type=int, help='Number of parallel steps', default=None)
    parser.add_argument('--Nsweep', type=int, help='Number of ranks in the sweeper', default=None)
    parser.add_argument('--Nspace', type=int, help='Number of ranks in space', default=None)
    parser.add_argument('--num_runs', type=int, help='Number of runs for statistics', default=5)
    parser.add_argument('--useGPU', type=cast_to_bool, help='Toggle for GPUs', default=True)
    parser.add_argument('--space_resolution', type=int, help='Resolution in space of the finest level')
    parser.add_argument('--Tend', type=float, help='Time to solve to')
    parser.add_argument('--problem', type=get_problem, help='Problem to run')
    parser.add_argument('--experiment', type=get_experiment, help='Experiment to run')
    parser.add_argument('--restart_idx', type=int, help='Restart from file by index')
    parser.add_argument('--procs', type=str_to_procs, help='Processes in steps/sweeper/space', default='1/1/1')
    parser.add_argument('--space_levels', type=int, help='Number of levels in space', default=None)
    # parser.add_argument('--time_levels', type=int, help='Number of levels in time', default = None)
    parser.add_argument(
        '--logger_level', type=int, help='Logger level on the first rank in space and in the sweeper', default='30'
    )

    kwargs = vars(parser.parse_args())

    for k, i in zip(['Nsteps', 'Nsweep', 'Nspace'], [0, 1, 2]):
        if kwargs[k] is None:
            kwargs[k] = kwargs.get(
                'procs',
                [
                    None,
                ]
                * 3,
            )[i]
    return kwargs


class RunAllenCahn(RunProblem):
    default_Tend = 2e-1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, imex=True)

    def get_default_description(self):
        from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

        description = super().get_default_description()

        description['step_params']['maxiter'] = 19

        description['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        description['sweeper_params']['num_nodes'] = 4
        description['sweeper_params']['QI'] = 'MIN-SR-S'
        description['sweeper_params']['QE'] = 'PIC'

        description['problem_params']['nvars'] = (2**10,) * 2
        description['problem_params']['init_type'] = 'circle_rand'
        description['problem_params']['L'] = 16
        description['problem_params']['spectral'] = False
        description['problem_params']['comm'] = self.comm_space
        description['problem_params']['eps'] = 0.01

        description['problem_class'] = allencahn_imex

        description['level_params']['dt'] = 0.5 * description['problem_params']['eps'] ** 2
        description['level_params']['restol'] = 1e-8

        return description

    def set_space_resolution(self, *args, **kwargs):
        super().set_space_resolution(*args, **kwargs)
        resolution = self.description['problem_params']['nvars']
        resolution = resolution if isinstance(resolution, list) else [resolution]
        self.description['problem_params']['L'] = (resolution[0][0] + 511) // 512

    @property
    def get_poly_adaptivity_default_params(self):
        defaults = super().get_poly_adaptivity_default_params
        defaults['e_tol'] = 1e-8
        defaults['dt_max'] = 0.9 * self.description['problem_params']['eps'] ** 2
        return defaults

    @staticmethod
    def plot(experiment, procs, idx, fig=None, **plotting_params):
        import matplotlib.pyplot as plt

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axs()

        plotting_args = {
            'vmin': 0,
            'vmax': 1,
            'rasterized': True,
            **plotting_params,
        }

        for n in range(procs[2]):
            solution = experiment.get_solution([procs[0], procs[1], n + 1], idx)
            x, y = experiment.get_grid([procs[0], procs[1], n + 1])
            ax.pcolormesh(x, y, solution['u'], **plotting_args)
        ax.set_aspect(1.0)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(f'$t$ = {solution["t"]:.2f}')
        return fig


class RunAllenCahnForcing(RunAllenCahn):
    # default_Tend = 1e-2

    def get_default_description(self):
        from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex_timeforcing

        description = super().get_default_description()
        description['problem_class'] = allencahn_imex_timeforcing
        return description


class RunAllenCahnAdaptivity(RunAllenCahn):
    default_Tend = 1.3e-1

    def get_default_description(self):
        from pySDC.projects.GPU.problem_classes.AllenCahn_MPIFFT import allencahn_imex_timeforcing_adaptivity

        description = super().get_default_description()
        description['problem_class'] = allencahn_imex_timeforcing_adaptivity
        description['problem_params']['time_freq'] = 1 / 2e-1 * 1.5
        # description['problem_params']['time_freq'] = 1 / 4e-1 * 1.5
        description['problem_params']['time_dep_strength'] = 5.0e-1
        return description


class RunAllenCahnSharpInterface(RunAllenCahn):
    default_Tend = 0.032

    def get_default_description(self):
        description = super().get_default_description()
        description['problem_params']['init_type'] = 'circle'
        return description

    def set_space_resolution(self, *args, **kwargs):
        super().set_space_resolution(*args, **kwargs)
        resolution = self.description['problem_params']['nvars']
        resolution = resolution if isinstance(resolution, list) else [resolution]
        self.description['problem_params']['L'] = 1
        self.description['problem_params']['eps'] = 0.04 * 128 / resolution[0][0]

    @classmethod
    def get_visualisation_hooks(cls):
        from pySDC.projects.GPU.problem_classes.AllenCahn_monitor import monitor_MPI

        return [monitor_MPI]


class RunSchroedinger(RunProblem):
    default_Tend = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, imex=True)

    def get_default_description(self):
        from pySDC.implementations.problem_classes.NonlinearSchroedinger_MPIFFT import nonlinearschroedinger_imex

        description = super().get_default_description()

        description['step_params']['maxiter'] = 19

        description['level_params']['dt'] = 1e-2
        description['level_params']['restol'] = 1e-8

        description['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        description['sweeper_params']['num_nodes'] = 4
        description['sweeper_params']['QI'] = 'MIN-SR-FLEX'
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


class RunBrusselator(RunProblem):
    default_Tend = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, imex=True)

    def get_default_description(self):
        from pySDC.implementations.problem_classes.Brusselator import Brusselator

        description = super().get_default_description()

        description['step_params']['maxiter'] = 9

        description['level_params']['dt'] = 1e-2
        description['level_params']['restol'] = 1e-8

        description['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        description['sweeper_params']['num_nodes'] = 4
        description['sweeper_params']['QI'] = 'MIN-SR-S'
        description['sweeper_params']['QE'] = 'PIC'

        description['problem_params']['nvars'] = (2**13,) * 2
        description['problem_params']['comm'] = self.comm_space

        description['problem_class'] = Brusselator

        return description

    @property
    def get_poly_adaptivity_default_params(self):
        defaults = super().get_poly_adaptivity_default_params
        defaults['e_tol'] = 1e-7
        return defaults


class RunGS(RunProblem):
    default_Tend = 10000
    Du = 2e-5
    Dv = 1e-5
    A = 0.04
    B = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, imex=True)

    def get_default_description(self):
        from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_diffusion

        description = super().get_default_description()

        description['step_params']['maxiter'] = 29

        description['level_params']['dt'] = 1e1
        description['level_params']['restol'] = 1e-8

        description['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        description['sweeper_params']['num_nodes'] = 4
        description['sweeper_params']['QI'] = 'MIN-SR-S'
        description['sweeper_params']['QE'] = 'PIC'

        description['problem_params']['nvars'] = (2**13,) * 2
        description['problem_params']['comm'] = self.comm_space
        description['problem_params']['Du'] = self.Du
        description['problem_params']['Dv'] = self.Dv
        description['problem_params']['A'] = self.A
        description['problem_params']['B'] = self.B
        description['problem_params']['num_blobs'] = 256
        description['problem_params']['init_type'] = 'rectangles'

        description['problem_class'] = grayscott_imex_diffusion

        return description

    @property
    def get_poly_adaptivity_default_params(self):
        defaults = super().get_poly_adaptivity_default_params
        defaults['e_tol'] = 1e-5
        defaults['dt_max'] = 20
        return defaults

    @staticmethod
    def plot(experiment, procs, idx, fig=None, **plotting_params):
        import matplotlib.pyplot as plt

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axs()

        plotting_args = {
            'vmin': 0,
            'vmax': 1,
            'rasterized': True,
            **plotting_params,
        }

        for n in range(procs[2]):
            solution = experiment.get_solution([procs[0], procs[1], n + 1], idx)
            x, y = experiment.get_grid([procs[0], procs[1], n + 1])
            ax.pcolormesh(x, y, solution['u'][0], **plotting_args)
        ax.set_aspect(1.0)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(f'$t$ = {solution["t"]:.2f}')
        return fig


class RunGS_GoL(RunGS):
    default_Tend = 10000#0
    Du = 2e-5
    Dv = 1e-5
    A = 0.062
    B = 0.062 + 0.0609


class RunGS_Turing(RunGS):
    default_Tend = 100000
    Du = 2e-5
    Dv = 1e-5
    A = 0.060
    B = 0.062 + 0.060


class SingleGPUExperiment(Experiment):
    name = 'single_gpu'


class AdaptivityExperiment(Experiment):
    name = 'adaptivity'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prob.add_polynomial_adaptivity()


class PFASST(Experiment):
    name = 'PFASST'

    def __init__(self, **kwargs):
        from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

        controller_params = {'hook_class': LogGlobalErrorPostRun, **kwargs.get('custom_controller_params', {})}

        kwargs = {
            **kwargs,
            'custom_controller_params': controller_params,
        }
        kwargs['space_levels'] = 2 if kwargs['space_levels'] is None else kwargs['space_levels']

        super().__init__(**kwargs)
        self.prob.add_polynomial_adaptivity()


# class CUDASweeperExperiment(Experiment):
#     name = 'CUDA'
#     def __init__(self, **kwargs):
#         from pySDC.projects.GPU.sweepers.imex_CUDA import IMEX_CUDA, imex_1st_orderCUDA, IMEX_diag_streams
#         description = {
#                 'sweeper_class': IMEX_diag_streams,
#         }
#         super().__init__(custom_description=description, **kwargs)


class Visualisation(AdaptivityExperiment):
    name = 'visualisation'
    log_solution = LogToFileAfterXs
    log_grid = LogGrid

    def __init__(self, live_plotting=False, **kwargs):
        from pySDC.implementations.hooks.log_step_size import LogStepSize
        from pySDC.implementations.hooks.log_work import LogWork, LogSDCIterations
        from pySDC.projects.GPU.run_problems import get_comms
        from pySDC.projects.GPU.hooks.LogStats import LogStats

        self.comm_steps, self.comm_sweep, self.comm_space = get_comms(kwargs.get('num_procs', [1, 1, 1]))

        self.log_solution.time_increment = kwargs['problem'].default_Tend / 500

        # hooks
        hooks = [LogStepSize]
        if self.comm_sweep.rank == self.comm_sweep.size - 1:
            hooks += [self.log_solution, LogGrid, LogWork, LogSDCIterations]
        if live_plotting:
            from pySDC.implementations.hooks.live_plotting import PlotPostStep

            hooks += [PlotPostStep]
        hooks += kwargs['problem'].get_visualisation_hooks()
        kwargs['custom_controller_params'] = {'hook_class': hooks, 'logger_level': 15}
        super().__init__(**kwargs)

        self._stats_name = PathFormatter.complete_fname(name='stats', **self.path_args)
        self.prob.description['convergence_controllers'][LogStats] = {
            'hook': self.log_solution,
            'file_name': self._stats_name,
        }

    def run(self, Tend=None, restart_idx=None):
        import pickle

        u_init = None
        t0 = 0.0
        stats_restart = {}
        if restart_idx:
            # load solution
            _u_init = self.log_solution.load(restart_idx)
            u_init = _u_init['u']
            t0 = _u_init['t']
            self.prob.description['level_params']['dt'] = _u_init['dt']

            # load stats
            stats_path = (
                f'{self.log_solution.path}/{self._stats_name}_{self.log_solution.format_index(restart_idx)}.pickle'
            )
            with open(stats_path, 'rb') as file:
                stats_restart = pickle.load(file)

            self.log_solution.counter = restart_idx + 1
            self.log_solution.t_next_log = t0 + self.log_solution.time_increment

        _stats = self.prob.run(Tend=Tend, u_init=u_init, t0=t0)
        stats = {**_stats, **stats_restart}

        # from pySDC.helpers.stats_helper import get_sorted
        # data = {
        #     'dt': get_sorted(stats, type='dt', recomputed=False, comm=self.comm_steps),
        #     'restart': get_sorted(stats, type='restart', recomputed=None, comm=self.comm_steps),
        # }

        args = {key: value for key, value in self.path_args.items() if key not in ['index']}
        with open(
            PathFormatter.complete_fname(name='stats', format='pickle', base_path=f'{self.log_solution.path}', **args),
            'wb',
        ) as file:
            pickle.dump(stats, file)
        with open(
            PathFormatter.complete_fname(
                name='stats', format='pickle', base_path=f'{self.log_solution.path}', **self.path_args
            ),
            'wb',
        ) as file:
            pickle.dump(stats, file)

    def plot(self, ax, ranks, idx):
        data = self.get_solution(ranks, idx)
        grid = self.get_grid(ranks)

        ax.pcolormesh(grid[0], grid[1], data['u'][0], vmin=0, vmax=7.0)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        ax.set_aspect(1.0)
        ax.set_title(data['t'])
        del data
        del grid


def get_problem(name):
    probs = {
        'Schroedinger': RunSchroedinger,
        'AC': RunAllenCahn,
        'ACT': RunAllenCahnForcing,
        'Brusselator': RunBrusselator,
        'GS': RunGS,
        'GSGoL': RunGS_GoL,
        'GST': RunGS_Turing,
        'ACA': RunAllenCahnAdaptivity,
        'ACI': RunAllenCahnSharpInterface,
    }
    return probs[name]


def get_experiment(name):
    ex = {
        'singleGPU': SingleGPUExperiment,
        'adaptivity': AdaptivityExperiment,
        # 'CUDA': CUDASweeperExperiment,
        'visualisation': Visualisation,
        'PFASST': PFASST,
    }
    return ex[name]
