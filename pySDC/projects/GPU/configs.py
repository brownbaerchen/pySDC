from pySDC.projects.GPU.run_problems import RunProblem
from pySDC.projects.GPU.run_problems import Experiment


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
        'problem': get_problem,
        'experiment': get_experiment,
    }

    args = {}
    for me in sys.argv[1:]:
        for key, cast in allowed_args.items():
            if key in me:
                args[key] = cast(me[len(key) + 1 :])

    return args


class RunAllenCahn(RunProblem):
    default_Tend = 2e-1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, imex=True)

    def get_default_description(self):
        from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

        description = super().get_default_description()

        description['step_params']['maxiter'] = 5

        description['level_params']['dt'] = 1e-4
        description['level_params']['restol'] = 1e-8

        description['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        description['sweeper_params']['num_nodes'] = 4
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


class RunAllenCahnForcing(RunAllenCahn):
    default_Tend = 1e-2

    def get_default_description(self):
        from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex_timeforcing

        description = super().get_default_description()
        description['problem_class'] = allencahn_imex_timeforcing
        return description


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
        defaults['interpolate_between_restarts'] = False
        return defaults


class SingleGPUExperiment(Experiment):
    name = 'single_gpu'


class AdaptivityExperiment(Experiment):
    name = 'adaptivity'

    def __init__(self, **kwargs):
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

    def __init__(self, **kwargs):
        from pySDC.implementations.hooks.log_solution import LogToFileAfterXs
        from pySDC.projects.GPU.run_problems import get_comms
        from pySDC.projects.GPU.hooks.LogGrid import LogGrid
        from pySDC.implementations.hooks.log_step_size import LogStepSize

        self.comm_steps, self.comm_sweep, self.comm_space = get_comms(kwargs.get('num_procs', [1, 1, 1]))
        rank_path = f'{self.comm_steps.rank}_{self.comm_sweep.rank}_{self.comm_space.rank}'

        prob_name = kwargs['problem'].__name__

        LogToFileAfterXs.path = './simulation_output'
        LogToFileAfterXs.file_name = f'solution_{prob_name}_{rank_path}'
        LogToFileAfterXs.time_increment = kwargs['problem'].default_Tend / 100
        if kwargs['useGPU']:
            import numpy as np

            LogToFileAfterXs.process_solution = lambda L: {'t': L.time + L.dt, 'u': L.uend.get().view(np.ndarray)}
        self.logger_hook = LogToFileAfterXs

        LogGrid.file_logger = LogToFileAfterXs
        LogGrid.file_name = f'grid_{prob_name}_{rank_path}'
        self.log_grid = LogGrid

        if self.comm_sweep.rank == self.comm_sweep.size - 1:
            controller_params = {'hook_class': [LogToFileAfterXs, LogGrid, LogStepSize], 'logger_level': 15}
        else:
            controller_params = {'hook_class': [LogStepSize], 'logger_level': 15}
        super().__init__(custom_controller_params=controller_params, **kwargs)

    def run(self, Tend=None):
        import pickle
        from pySDC.helpers.stats_helper import get_sorted

        stats = self.prob.run(Tend=Tend)

        data = {'dt': get_sorted(stats, type='dt', recomputed=False, comm=self.comm_steps)}

        if self.comm_steps.rank > 0 or self.comm_sweep.rank > 0 or self.comm_space.rank > 0:
            return None

        with open(f'{self.logger_hook.path}/{type(self.prob).__name__}_stats.pickle', 'wb') as file:
            pickle.dump(data, file)

    def get_grid(self, ranks):
        self.log_grid.file_name = f'grid_{type(self.prob).__name__}_{ranks[0]}_{ranks[1]}_{ranks[2]}'
        return self.log_grid.load()

    def get_solution(self, ranks, idx):
        self.logger_hook.file_name = f'solution_{type(self.prob).__name__}_{ranks[0]}_{ranks[1]}_{ranks[2]}'
        return self.logger_hook.load(idx)

    def plot(self, ax, ranks, idx):
        data = self.get_solution(ranks, idx)
        grid = self.get_grid(ranks)

        ax.pcolormesh(grid[0], grid[1], data['u'][0], vmin=0, vmax=7.0)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        ax.set_aspect(1.0)
        ax.set_title(data['t'])


def get_problem(name):
    probs = {
        'Schroedinger': RunSchroedinger,
        'AC': RunAllenCahn,
        'ACT': RunAllenCahnForcing,
        'Brusselator': RunBrusselator,
    }
    return probs[name]


def get_experiment(name):
    ex = {
        'singleGPU': SingleGPUExperiment,
        'adaptivity': AdaptivityExperiment,
        # 'CUDA': CUDASweeperExperiment,
        'visualisation': Visualisation,
    }
    return ex[name]
