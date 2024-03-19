import numpy as np
from mpi4py import MPI
from pySDC.projects.Resilience.strategies import merge_descriptions
from pySDC.helpers.stats_helper import get_sorted
import pickle
import matplotlib.pyplot as plt


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

    def record_timing(self, num_runs=5, name=''):
        errors = []
        times = []

        for _ in range(num_runs):
            stats = self.run()

            errors += [max(me[1] for me in get_sorted(stats, type='e_global_post_run'))]
            times += [max(me[1] for me in get_sorted(stats, type='timing_run'))]
            if self.comm_sweep.rank == 0 and self.comm_space.rank == 0 and self.comm_steps.rank == 0:
                print(f'Needed {times[-1]:.2e}s with error {errors[-1]:.2e}', flush=True)

        if self.comm_sweep.rank == 0 and self.comm_space.rank == 0 and self.comm_steps.rank == 0:
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
        return f'{base_path}/{prob}_{procs}_{gpu}{name}.pickle'

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
    }

    args = {}
    for me in sys.argv[1:]:
        for key, cast in allowed_args.items():
            if key in me:
                args[key] = cast(me[len(key) + 1 :])

    return args


def single_gpu_experiments(problem, num_procs, useGPU=True, num_runs=5, space_resolution=2**13):
    description = {'problem_params': {'nvars': (space_resolution,) * 2}}
    prob = problem(num_procs=num_procs, useGPU=useGPU, custom_description=description)
    prob.record_timing(num_runs=num_runs, name=f'single_gpu_{space_resolution:d}')


def plot_single_gpu_experiments(problem):
    useGPU = True
    num_procs = [1, 1, 4]

    procs_parallel_sweeper = [[1, 4, 1], [1, 4, 4], [1, 4, 8], [1, 4, 16], [1, 4, 32], [1, 4, 64]]
    procs_serial_sweeper = [[1, 1, 4], [1, 1, 16], [1, 1, 32], [1, 1, 64], [1, 1, 128], [1, 1, 256]]

    procs_parallel_sweeper_CPU = [[1, 4, 384]]
    procs_serial_sweeper_CPU = [[1, 1, 384], [1, 1, 1536]]

    def get_multiple_data(num_procs_list, useGPU):
        times = {}
        for num_procs in num_procs_list:
            prob = problem(num_procs=num_procs, useGPU=useGPU)
            data = prob.get_data('single_gpu')
            fac = 4 if useGPU else 96
            times[np.prod(num_procs) / fac] = data['times']
        return times

    times_parallel_sweeper = get_multiple_data(procs_parallel_sweeper, True)
    times_serial_sweeper = get_multiple_data(procs_serial_sweeper, True)
    times_parallel_sweeper_CPU = get_multiple_data(procs_parallel_sweeper_CPU, False)
    times_serial_sweeper_CPU = get_multiple_data(procs_serial_sweeper_CPU, False)

    print(times_parallel_sweeper)
    print(times_serial_sweeper)

    fig, ax = plt.subplots()
    for times, labels in zip([times_parallel_sweeper, times_serial_sweeper], ['parallel sweeper', 'serial sweeper']):
        ax.loglog([me for me in times.keys()], [me for me in times.values()], label=f'{labels} GPU', marker='x')
    # for times, labels in zip([times_parallel_sweeper_CPU, times_serial_sweeper_CPU], ['parallel sweeper', 'serial sweeper']):
    #     ax.plot([me for me in times.keys()], [me for me in times.values()], label=f'{labels} CPU', marker='x')
    ax.legend(frameon=False)
    ax.set_xlabel('Nodes')
    ax.set_ylabel('wall time / s')
    plt.show()


if __name__ == '__main__':
    problem = RunSchroedinger
    args = parse_args()

    num_procs = [args.get(key, 1) for key in ['Nsteps', 'Nsweep', 'Nspace']]

    # single_gpu_experiments(
    #     problem,
    #     num_procs=num_procs,
    #     num_runs=args.get('num_runs', 5),
    #     useGPU=args.get('useGPU', True),
    #     space_resolution=args.get('space_resolution', 2**13),
    # )
    plot_single_gpu_experiments(problem)
