import numpy as np
from mpi4py import MPI
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.GPU.utils import PathFormatter, DummyLogger
import pickle


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
        u_init=None,
        space_levels=1,
        time_levels=1,
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
        self.space_levels = space_levels
        self.time_levels = time_levels

        custom_description = {} if custom_description is None else custom_description
        custom_controller_params = {} if custom_controller_params is None else custom_controller_params

        self.useGPU = useGPU
        self.num_procs = num_procs
        self.comm_steps, self.comm_sweep, self.comm_space = get_comms(num_procs, comm_world=comm_world)
        self.get_description(custom_description, imex)
        self.get_controller_params(custom_controller_params)
        if space_resolution:
            self.set_space_resolution(space_resolution, levels=space_levels)

        if self.comm_sweep.size > 1:
            from pySDC.implementations.transfer_classes.BaseTransferMPI import base_transfer_MPI

            self.description['base_transfer_class'] = base_transfer_MPI

        self.u_init = u_init

    @classmethod
    def get_visualisation_hooks(cls):
        return []

    def get_default_description(self):
        from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingMPI

        description = {
            'step_params': {},
            'level_params': {},
            'sweeper_params': {},
            'sweeper_class': None,
            'problem_params': {},
            'problem_class': None,
            'convergence_controllers': {BasicRestartingMPI: {'max_restarts': 29}},
            'space_transfer_class': fft_to_fft,
            'space_transfer_params': {'periodic': True},
        }
        description['problem_params']['useGPU'] = self.useGPU
        return description

    def set_space_resolution(self, resolution, levels=1):
        levels = 1 if levels is None else levels
        current_resolution = self.get_space_resolution()
        self.description['problem_params']['nvars'] = [
            (int(resolution // 2**i),) * len(current_resolution) for i in range(levels)
        ]

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
            'interpolate_between_restarts': False,
            'abort_at_growing_residual': True,
            'dt_min': 1e-8,
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
        controller_params = {'logger_level': 30, 'mssdc_jac': False, 'all_to_done': True, 'hook_class': []}
        return controller_params

    def get_controller_params(self, custom_controller_params):
        self.controller_params = {**self.get_default_controller_params(), **custom_controller_params}

        # prevent duplicate logging
        if self.comm_space.rank > 0 or self.comm_sweep.rank > 0:
            self.controller_params['logger_level'] = 30

    def run(self, Tend=None, u_init=None, t0=0):
        from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

        Tend = self.default_Tend if Tend is None else Tend
        controller = controller_MPI(
            description=self.description, controller_params=self.controller_params, comm=self.comm_steps
        )
        prob = controller.S.levels[0].prob
        _u0 = prob.u_exact(0)
        if u_init is not None:
            _u0[:] = u_init[:]
        u_end, stats = controller.run(u0=_u0, t0=t0, Tend=Tend)

        return stats

    def print(self, *args):
        if self.comm_steps.rank == self.comm_steps.size - 1 and self.comm_space.rank == 0 and self.comm_sweep.rank == 0:
            print(*args, flush=True)

    def record_timing(self, num_runs=5, name='', Tend=None):
        errors = []
        times = []

        for _ in range(num_runs):
            stats = self.run(Tend=Tend)

            errors += [max([-1] + [me[1] for me in get_sorted(stats, type='e_global_post_run')])]
            times += [max(me[1] for me in get_sorted(stats, type='timing_run'))]
            self.print(f'Needed {times[-1]:.2e}s with error {errors[-1]:.2e}')

        if (
            self.comm_sweep.rank == 0
            and self.comm_space.rank == 0
            and self.comm_steps.rank == self.comm_steps.size - 1
            and Tend is None
        ):
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
        space_resolution = f'{self.get_space_resolution()[0][0]:d}'
        levels = f'l{self.space_levels}x{self.time_levels}'
        return f'{base_path}/{prob}_{procs}_{gpu}{name}_{space_resolution}_{levels}.pickle'

    def get_data(self, name=''):
        with open(self.get_path(name=name), 'rb') as file:
            data = pickle.load(file)

        results = {}
        for key, values in data.items():
            results[key] = np.median(values)
            if np.std(values) / np.median(values) > 1e-2:
                print(f'Warning! Standard deviation too large in {key!r}!')
        return results

    @staticmethod
    def plot(experiment, procs, idx, fig=None, **plotting_params):
        raise NotImplementedError


class Experiment:
    name = None
    log_grid = DummyLogger
    log_solution = DummyLogger

    def __init__(
        self,
        problem,
        num_runs=5,
        custom_description=None,
        custom_controller_params=None,
        **kwargs,
    ):
        kwargs['problem'] = problem
        self.problem = problem
        self.num_runs = num_runs

        self.prob_args = {
            'num_procs': kwargs.get('num_procs', [1, 1, 1]),
            'useGPU': kwargs.get('useGPU', True),
            'custom_description': custom_description if custom_description else {},
            'space_resolution': kwargs.get('space_resolution', None),
            'custom_controller_params': custom_controller_params,
            'space_levels': kwargs.get('space_levels', 1),
        }

        self.prob = problem(**self.prob_args)

        self.comm_steps, self.comm_sweep, self.comm_space = get_comms(kwargs.get('num_procs', [1, 1, 1]))

        self.path_args = {**kwargs, 'num_procs': [self.comm_steps, self.comm_sweep, self.comm_space]}
        for key in ['restart_idx']:
            self.path_args.pop(key, None)

        self.log_grid.file_name = PathFormatter.complete_fname(name='grid', **self.path_args)
        self.log_grid.path = './simulation_output'
        self.log_solution.file_name = PathFormatter.complete_fname(name='solution', **self.path_args)
        self.log_solution.path = './simulation_output'

        if kwargs['useGPU']:
            self.log_solution.process_solution = lambda L: {
                't': L.time + L.dt,
                'dt': L.status.dt_new,
                'u': L.uend.get().view(np.ndarray),
            }
        else:
            self.log_solution.process_solution = lambda L: {
                't': L.time + L.dt,
                'dt': L.status.dt_new,
                'u': L.uend.view(np.ndarray),
            }

        self.log_grid.file_logger = self.log_solution

    def get_hook_fname_for_ranks(self, ranks):
        return f'{type(self.prob).__name__}_{ranks[0]}_{ranks[1]}_{ranks[2]}'

    def get_grid(self, ranks):
        _fname = f'{self.log_grid.file_name}'
        self.log_grid.file_name = PathFormatter.complete_fname(**{**self.path_args, 'num_procs': ranks, 'name': 'grid'})
        res = self.log_grid.load()
        self.log_grid.file_name = _fname
        return res

    def get_solution(self, ranks, idx):
        _fname = f'{self.log_solution.file_name}'
        self.log_solution.file_name = PathFormatter.complete_fname(
            **{**self.path_args, 'num_procs': ranks, 'name': 'solution'}
        )
        res = self.log_solution.load(idx)
        self.log_solution.file_name = _fname
        return res

    def restart_from_file(self, ranks, idx):
        self.prob.u_init = self.get_solution(ranks, idx)

    def run(self, Tend=None, **kwargs):
        self.prob.record_timing(num_runs=self.num_runs, name=self.name, Tend=Tend)


if __name__ == '__main__':
    from pySDC.projects.GPU.configs import AdaptivityExperiment, RunAllenCahn, parse_args

    args = parse_args()

    num_procs = [args.get(key, 1) for key in ['Nsteps', 'Nsweep', 'Nspace']]

    custom_controller_params = {
        'logger_level': args.get('logger_level', None),
    }

    kwargs = {
        'num_procs': num_procs,
        'num_runs': args.get('num_runs', 5),
        'useGPU': args.get('useGPU', True),
        'space_resolution': args.get('space_resolution', 2**13),
        'problem': args.get('problem', RunAllenCahn),
        'custom_controller_params': custom_controller_params,
        'space_levels': args['space_levels'],
    }
    experiment = args.get('experiment', AdaptivityExperiment)(**kwargs)

    run_args = {
        'restart_idx': args.get('restart_idx', None),
        'Tend': args.get('Tend', None),
    }
    experiment.run(**run_args)
