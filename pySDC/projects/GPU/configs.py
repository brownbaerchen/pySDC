def get_config(args):
    name = args['config']
    if name == 'RBC':
        return RayleighBenardRegular(args)
    elif name == 'RBC_dt':
        return RayleighBenard_dt_adaptivity(args)
    elif name == 'RBC_k':
        return RayleighBenard_k_adaptivity(args)
    elif name == 'RBC_dt_k':
        return RayleighBenard_dt_k_adaptivity(args)
    elif name == 'RBC_HR':
        return RayleighBenardHighResolution(args)
    elif name == 'RBC_dt_HR':
        return RayleighBenard_dt_adaptivity_high_res(args)
    elif name == 'RBC_RK':
        return RayleighBenardRK(args)
    elif name == 'RBC_dedalus':
        return RayleighBenardDedalusComp(args)
    elif name == 'RBC_fast':
        return RayleighBenard_fast(args)
    elif name == 'RBC_Tibo':
        return RayleighBenard_Thibaut(args)
    elif name == 'RBC_TiboHR':
        return RayleighBenard_Thibaut_HighRes(args)
    else:
        raise NotImplementedError(f'There is no configuration called {name!r}!')


def get_comms(n_procs_list, comm_world=None, _comm=None, _tot_rank=0, _rank=None, useGPU=False):
    from mpi4py import MPI

    comm_world = MPI.COMM_WORLD if comm_world is None else comm_world
    _comm = comm_world if _comm is None else _comm
    _rank = comm_world.rank if _rank is None else _rank

    if len(n_procs_list) > 0:
        color = _tot_rank + _rank // n_procs_list[0]
        new_comm = comm_world.Split(color)

        if useGPU:
            import cupy_backends

            try:
                import cupy
                from pySDC.helpers.NCCL_communicator import NCCLComm

                new_comm = NCCLComm(new_comm)
            except (
                ImportError,
                cupy_backends.cuda.api.runtime.CUDARuntimeError,
                cupy_backends.cuda.libs.nccl.NcclError,
            ):
                print('Warning: Communicator is MPI instead of NCCL in spite of using GPUs!')

        return [new_comm] + get_comms(
            n_procs_list[1:],
            comm_world,
            _comm=new_comm,
            _tot_rank=_tot_rank + _comm.size * new_comm.rank,
            _rank=_comm.rank // new_comm.size,
            useGPU=useGPU,
        )
    else:
        return []


class Config(object):
    sweeper_type = None
    name = None
    experiment_name = 'regular'
    Tend = None

    def __init__(self, args, comm_world=None):
        from mpi4py import MPI

        self.args = args
        self.comm_world = MPI.COMM_WORLD if comm_world is None else comm_world
        self.n_procs_list = args["procs"]
        self.comms = get_comms(n_procs_list=self.n_procs_list, useGPU=args['useGPU'])
        self.ranks = [me.rank for me in self.comms]

    def get_description(self, *args, MPIsweeper=False, useGPU=False, **kwargs):
        description = {}
        description['problem_class'] = None
        description['problem_params'] = {'useGPU': useGPU, 'comm': self.comms[2]}
        description['sweeper_class'] = self.get_sweeper(useMPI=MPIsweeper)
        description['sweeper_params'] = {'initial_guess': 'copy'}
        description['level_params'] = {}
        description['step_params'] = {}
        description['convergence_controllers'] = {}

        if MPIsweeper:
            description['sweeper_params']['comm'] = self.comms[1]
        return description

    def get_controller_params(self, *args, logger_level=15, **kwargs):
        from pySDC.implementations.hooks.log_work import LogWork

        controller_params = {}
        controller_params['logger_level'] = logger_level if self.comm_world.rank == 0 else 40
        controller_params['hook_class'] = [LogWork] + self.get_LogToFile()
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
        return f'{self.name}-{type(self).__name__}{self.args_to_str()}-{ranks[0]}-{ranks[2]}'

    def args_to_str(self, args=None):
        args = self.args if args is None else args
        name = ''

        name = f'{name}-useGPU_{args["useGPU"]}'
        name = f'{name}-procs_{args["procs"][0]}_{args["procs"][1]}_{args["procs"][2]}'
        return name

    def plot(self, P, idx):
        raise NotImplementedError

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if restart_idx > 0:
            LogToFile = self.get_LogToFile()[0]
            file = LogToFile.load(restart_idx)
            LogToFile.counter = restart_idx
            u0 = P.u_init
            if P.spectral_space:
                u0[...] = P.transform(file['u'])
            else:
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
        LogToFile.file_name = f'{self.get_path(ranks=ranks)}-solution'
        LogToFile.time_increment = 1e-1

        def process_solution(L):
            P = L.prob

            if P.spectral_space:
                uend = P.itransform(L.uend)

            if P.useGPU:
                return {
                    't': L.time + L.dt,
                    'u': uend.get().view(np.ndarray),
                    'X': L.prob.X.get().view(np.ndarray),
                    'Z': L.prob.Z.get().view(np.ndarray),
                    'vorticity': L.prob.compute_vorticity(L.uend).get().view(np.ndarray),
                    # 'divergence': L.uend.xp.log10(L.uend.xp.abs(L.prob.eval_f(L.uend).impl[L.prob.index('p')])).get(),
                }
            else:
                return {
                    't': L.time + L.dt,
                    'u': uend.view(np.ndarray),
                    'X': L.prob.X.view(np.ndarray),
                    'Z': L.prob.Z.view(np.ndarray),
                    'vorticity': L.prob.compute_vorticity(L.uend).view(np.ndarray),
                    # 'divergence': L.uend.xp.log10(L.uend.xp.abs(L.prob.eval_f(L.uend).impl[L.prob.index('p')])),
                }

        def logging_condition(L):
            sweep = L.sweep
            if hasattr(sweep, 'comm'):
                if sweep.comm.rank == sweep.comm.size - 1:
                    return True
                else:
                    return False
            else:
                return True

        LogToFile.process_solution = process_solution
        LogToFile.logging_condition = logging_condition
        return [LogToFile]

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )
        from pySDC.implementations.hooks.log_step_size import LogStepSize

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] += [LogAnalysisVariables, LogStepSize]
        return controller_params

    def get_description(self, *args, MPIsweeper=False, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            RayleighBenard,
            CFLLimit,
        )
        from pySDC.implementations.problem_classes.generic_spectral import (
            compute_residual_DAE,
            compute_residual_DAE_MPI,
        )
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter

        desc = super().get_description(*args, MPIsweeper=MPIsweeper, **kwargs)

        if MPIsweeper:
            desc['sweeper_class'].compute_residual = compute_residual_DAE_MPI
        else:
            desc['sweeper_class'].compute_residual = compute_residual_DAE

        desc['level_params']['dt'] = 0.1
        desc['level_params']['restol'] = 1e-7

        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 0.8}
        desc['convergence_controllers'][StepSizeSlopeLimiter] = {'dt_rel_min_slope': 0.1}

        desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        desc['sweeper_params']['num_nodes'] = 2
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'

        desc['problem_params']['Rayleigh'] = 2e6
        desc['problem_params']['nx'] = 2**8
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
            ranks = self.ranks[:-1] + [rank]
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


class RayleighBenard_k_adaptivity(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit, SpaceAdaptivity

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 0.8}
        desc['convergence_controllers'][SpaceAdaptivity] = {'nz_max': 512}
        desc['level_params']['restol'] = 1e-7
        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['step_params']['maxiter'] = 12

        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = [
            me for me in controller_params['hook_class'] if me is not LogAnalysisVariables
        ]
        return controller_params


class RayleighBenard_dt_k_adaptivity(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit, SpaceAdaptivity

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][AdaptivityPolynomialError] = {
            'e_tol': 1e-2,
            'abort_at_growing_residual': False,
            'interpolate_between_restarts': False,
            # 'dt_min': 1e-3,
        }
        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 1.0}
        desc['convergence_controllers'][SpaceAdaptivity] = {'nz_max': 256}
        desc['level_params']['restol'] = 1e-7
        # desc['level_params']['e_tol'] = 1e-10
        desc['sweeper_params']['num_nodes'] = 3
        desc['step_params']['maxiter'] = 12

        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = [
            me for me in controller_params['hook_class'] if me is not LogAnalysisVariables
        ]
        return controller_params


class RayleighBenard_dt_adaptivity(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-4, 'dt_rel_min_slope': 0.1}
        desc['convergence_controllers'].pop(CFLLimit)
        desc['level_params']['restol'] = -1
        desc['sweeper_params']['num_nodes'] = 3
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['step_params']['maxiter'] = 5
        return desc


class RayleighBenard_dt_adaptivity_high_res(RayleighBenard_dt_adaptivity):
    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)

        desc['problem_params']['nx'] = 2**10
        desc['problem_params']['nz'] = 2**8

        desc['sweeper_params']['num_nodes'] = 3
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['step_params']['maxiter'] = 5
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        return desc


class RayleighBenard_fast(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['problem_params']['nx'] = 2**8
        desc['problem_params']['nz'] = 2**6
        desc['level_params']['restol'] = -1
        desc['sweeper_params']['num_nodes'] = 2
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['step_params']['maxiter'] = 3
        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 0.3}
        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = [
            me for me in controller_params['hook_class'] if me is not LogAnalysisVariables
        ]
        return controller_params


class RayleighBenardHighResolution(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)

        desc['sweeper_params']['num_nodes'] = 4

        desc['problem_params']['Rayleigh'] = 2e6
        desc['problem_params']['nx'] = 2**10
        desc['problem_params']['nz'] = 2**8

        desc['step_params']['maxiter'] = 4

        return desc


class RayleighBenard_Thibaut(RayleighBenardRegular):
    Tend = 1

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'].pop(CFLLimit)
        desc['level_params']['restol'] = -1
        desc['level_params']['dt'] = 2e-2 / 4
        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-FLEX'
        desc['sweeper_params']['node_type'] = 'LEGENDRE'
        desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['step_params']['maxiter'] = 4
        return desc

    def get_controller_params(self, *args, **kwargs):
        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = []
        return controller_params


class RayleighBenard_Thibaut_HighRes(RayleighBenard_Thibaut):
    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)

        desc['problem_params']['nx'] = 2**9
        desc['problem_params']['nz'] = 2**7
        return desc


class RayleighBenardRK(RayleighBenardRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK222

        desc = super().get_description(*args, **kwargs)

        desc['sweeper_class'] = ARK222

        desc['step_params']['maxiter'] = 1
        # desc['level_params']['dt'] = 0.1

        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc['convergence_controllers'][CFLLimit] = {'dt_max': 0.1, 'dt_min': 1e-6, 'cfl': 0.2}
        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard import (
            LogAnalysisVariables,
        )

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] = [
            me for me in controller_params['hook_class'] if me is not LogAnalysisVariables
        ]
        return controller_params


class RayleighBenardDedalusComp(RayleighBenardRK):
    Tend = 150

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK222

        desc = super().get_description(*args, **kwargs)

        desc['sweeper_class'] = ARK222

        desc['step_params']['maxiter'] = 1
        desc['level_params']['dt'] = 5e-3

        from pySDC.implementations.problem_classes.RayleighBenard import CFLLimit

        desc['convergence_controllers'].pop(CFLLimit)
        return desc


# class RayleighBenardDedalusComp(RayleighBenardRK):
#     Tend = 150
#     # def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
#     #     args = {'procs': self.n_procs_list, 'comm_world': self.comm_world}
#     #     regular = RayleighBenardRegular(n_procs_list=self.n_procs_list, comm_world=self.comm_world)
#     #     u, t = regular.get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)
#     #     type(self).Tend = t + self.t_inc
#
#     #     return u, t
