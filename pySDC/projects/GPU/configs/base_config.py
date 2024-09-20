def get_config(args):
    name = args['config']
    if name[:3] == 'RBC':
        from pySDC.projects.GPU.configs.RBC_configs import get_config
        return get_config(args)
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
