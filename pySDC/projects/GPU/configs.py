def get_config(name, n_procs_list):
    if name == 'RBC':
        return RayleighBenardRegular(n_procs_list=n_procs_list)
    else:
        raise NotImplementedError(f'There is no configuration called {name!r}!')


def get_comms(n_procs_list, comm_world=None, _comm=None, _tot_rank=0, _rank=None):
    from mpi4py import MPI

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


class Config(object):
    sweeper_type = None
    name = None
    experiment_name = 'regular'
    Tend = None

    def __init__(self, n_procs_list, comm_world=None):
        from mpi4py import MPI

        self.comm_world = MPI.COMM_WORLD if comm_world is None else comm_world
        self.n_procs_list = n_procs_list
        self.comms = get_comms(n_procs_list=n_procs_list)
        self.ranks = [me.rank for me in self.comms]

    def get_description(self, *args, MPIsweeper=False, **kwargs):
        description = {}
        description['problem_class'] = None
        description['problem_params'] = {}
        description['sweeper_class'] = self.get_sweeper(useMPI=MPIsweeper)
        description['sweeper_params'] = {}
        description['level_params'] = {'comm': self.comms[2]}
        description['step_params'] = {}
        description['convergence_controllers'] = {}

        if MPIsweeper:
            description['sweeper_params']['comm'] = self.comms[1]
        return description

    def get_controller_params(self, *args, logger_level=15, **kwargs):
        controller_params = {}
        controller_params['logger_level'] = logger_level if self.comm_world.rank == 0 else 40
        controller_params['hook_class'] = [] + self.get_LogToFile()
        controller_params['mssdc_jac'] = False
        return controller_params

    def get_sweeper(self, useMPI):
        if useMPI and self.sweeper_type == 'IMEX':
            from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI as sweeper
        elif useMPI and self.sweeper_type == 'generic_implicit':
            from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper
        elif not useMPI and self.sweeper_type == 'IMEX':
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper
        else:
            raise NotImplementedError

        return sweeper

    def get_path(self, *args, ranks=None, **kwargs):
        ranks = self.ranks if ranks is None else ranks
        return f'{self.name}-{type(self).__name__}-{ranks[0]}-{ranks[2]}'

    def plot(self, P, idx):
        raise NotImplementedError

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if restart_idx > 0:
            LogToFile = self.get_LogToFile()[0]
            file = LogToFile.load(restart_idx)
            LogToFile.counter = restart_idx
            u0 = P.u_init
            u0[...] = file['u']
            return u0, file['t']
        else:
            raise NotImplementedError

    def get_LogToFile(self):
        return []


class RayleighBenardRegular(Config):
    sweeper_type = 'IMEX'
    name = 'RBC'
    Tend = 50

    def get_LogToFile(self, ranks=None):
        import numpy as np
        from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile

        LogToFile.path = './data/'
        LogToFile.file_name = f'{self.get_path(ranks=ranks)}-solution.pickle'
        LogToFile.time_increment = 1e-1

        LogToFile.process_solution = lambda L: {
            't': L.time + L.dt,
            'u': L.uend.view(np.ndarray),
            'X': L.prob.X.view(np.ndarray),
            'Z': L.prob.Z.view(np.ndarray),
            'vorticity': L.prob.compute_vorticity(L.uend).view(np.ndarray),
            'divergence': L.uend.xp.log10(L.uend.xp.abs(L.prob.eval_f(L.uend).impl[L.prob.index('p')])),
        }
        return [LogToFile]

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )
        from pySDC.implementations.hooks.log_step_size import LogStepSize

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] += [LogAnalysisVariables, LogStepSize]
        return controller_params

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            RayleighBenard,
            CFLLimit,
        )
        from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE

        desc = super().get_description(*args, **kwargs)

        desc['sweeper_class'].compute_residual = compute_residual_DAE

        desc['level_params']['dt'] = 0.1
        desc['level_params']['restol'] = 1e-7

        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 0.4}

        desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        desc['sweeper_params']['num_nodes'] = 2
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'

        desc['problem_params']['Rayleigh'] = 2e6
        desc['problem_params']['nx'] = 2**8 + 1
        desc['problem_params']['nz'] = 2**6
        desc['problem_params']['dealiasing'] = 3 / 2

        desc['step_params']['maxiter'] = 3

        desc['problem_class'] = RayleighBenard

        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if restart_idx > 0:
            return super().get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)
        else:
            return P.u_exact(t=0, seed=self.comm_world.rank, noise_level=1e-3), 0

    def plot(self, P, idx, n_procs_list, quantitiy='T', quantitiy2='vorticity'):
        import numpy as np

        cmaps = {'vorticity': 'bwr', 'p': 'bwr'}

        fig = P.get_fig()
        cax = P.cax
        axs = fig.get_axes()

        buffer = {}
        vmin = {quantitiy: np.inf, quantitiy2: np.inf}
        vmax = {quantitiy: -np.inf, quantitiy2: -np.inf}

        for rank in range(n_procs_list[2]):
            ranks = [me for me in self.ranks[:-1]] + [rank]
            LogToFile = self.get_LogToFile(ranks=ranks)[0]

            buffer[f'u-{rank}'] = LogToFile.load(idx)

            vmin[quantitiy2] = min([vmin[quantitiy2], buffer[f'u-{rank}'][quantitiy2].real.min()])
            vmax[quantitiy2] = max([vmax[quantitiy2], buffer[f'u-{rank}'][quantitiy2].real.max()])
            vmin[quantitiy] = min([vmin[quantitiy], buffer[f'u-{rank}']['u'][P.index(quantitiy)].real.min()])
            vmax[quantitiy] = max([vmax[quantitiy], buffer[f'u-{rank}']['u'][P.index(quantitiy)].real.max()])

        for rank in range(n_procs_list[2]):

            im = axs[1].pcolormesh(
                buffer[f'u-{rank}']['X'],
                buffer[f'u-{rank}']['Z'],
                buffer[f'u-{rank}'][quantitiy2].real,
                vmin=-vmax[quantitiy2] if cmaps.get(quantitiy2, None) in ['bwr'] else vmin[quantitiy2],
                vmax=vmax[quantitiy2],
                cmap=cmaps.get(quantitiy2, None),
            )
            fig.colorbar(im, cax[1])
            im = axs[0].pcolormesh(
                buffer[f'u-{rank}']['X'],
                buffer[f'u-{rank}']['Z'],
                buffer[f'u-{rank}']['u'][P.index(quantitiy)].real,
                vmin=vmin[quantitiy],
                vmax=-vmin[quantitiy] if cmaps.get(quantitiy, None) in ['bwr', 'seismic'] else vmax[quantitiy],
                cmap=cmaps.get(quantitiy, 'plasma'),
            )
            fig.colorbar(im, cax[0])
            axs[0].set_title(f't={buffer[f"u-{rank}"]["t"]:.2f}')
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('z')
            axs[0].set_aspect(1.0)
            axs[1].set_aspect(1.0)
        return fig
