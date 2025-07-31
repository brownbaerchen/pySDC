from pySDC.projects.GPU.configs.base_config import Config


def get_config(args):
    name = args['config']
    if name == 'RBC3D':
        return RayleighBenard3DRegular(args)
    elif name in globals().keys():
        return globals()[name](args)
    else:
        raise NotImplementedError(f'There is no configuration called {name!r}!')


class RayleighBenard3DRegular(Config):
    sweeper_type = 'IMEX'
    Tend = 50

    def get_file_name(self):
        res = self.args['res']
        return f'{self.base_path}/data/{type(self).__name__}-res{res}.pySDC'

    def get_LogToFile(self, *args, **kwargs):
        if self.comms[1].rank > 0:
            return None
        import numpy as np
        from pySDC.implementations.hooks.log_solution import LogToFile

        LogToFile.filename = self.get_file_name()
        LogToFile.time_increment = 5e-1
        LogToFile.allow_overwriting = True

        return LogToFile

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.hooks.log_step_size import LogStepSize

        controller_params = super().get_controller_params(*args, **kwargs)
        controller_params['hook_class'] += [LogStepSize]
        return controller_params

    def get_description(self, *args, MPIsweeper=False, res=-1, **kwargs):
        from pySDC.implementations.problem_classes.RayleighBenard3D import (
            RayleighBenard3D,
        )
        from pySDC.implementations.problem_classes.generic_spectral import (
            compute_residual_DAE,
            compute_residual_DAE_MPI,
        )
        from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter
        from pySDC.implementations.convergence_controller_classes.crash import StopAtNan

        desc = super().get_description(*args, MPIsweeper=MPIsweeper, **kwargs)

        if MPIsweeper:
            desc['sweeper_class'].compute_residual = compute_residual_DAE_MPI
        else:
            desc['sweeper_class'].compute_residual = compute_residual_DAE

        desc['level_params']['dt'] = 0.01
        desc['level_params']['restol'] = 1e-7

        desc['convergence_controllers'][StepSizeSlopeLimiter] = {'dt_rel_min_slope': 0.1}
        desc['convergence_controllers'][StopAtNan] = {}

        desc['sweeper_params']['quad_type'] = 'RADAU-RIGHT'
        desc['sweeper_params']['num_nodes'] = 2
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'

        desc['problem_params']['Rayleigh'] = 1e8
        desc['problem_params']['nx'] = 64 if res == -1 else res
        desc['problem_params']['ny'] = desc['problem_params']['nx']
        desc['problem_params']['nz'] = desc['problem_params']['nx']
        desc['problem_params']['Lx'] = 1
        desc['problem_params']['Ly'] = 1
        desc['problem_params']['Lz'] = 1
        desc['problem_params']['heterogeneous'] = True

        desc['step_params']['maxiter'] = 3

        desc['problem_class'] = RayleighBenard3D

        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):

        if restart_idx == 0:
            u0 = P.u_exact(t=0, seed=P.comm.rank, noise_level=1e-3)
            u0_with_pressure = P.solve_system(u0, 1e-9, u0)
            P.cached_factorizations.pop(1e-9)
            return u0_with_pressure, 0
        else:
            from pySDC.helpers.fieldsIO import FieldsIO

            P.setUpFieldsIO()
            outfile = FieldsIO.fromFile(self.get_file_name())

            t0, solution = outfile.readField(restart_idx)

            u0 = P.u_init

            if P.spectral_space:
                u0[...] = P.transform(solution)
            else:
                u0[...] = solution

            return u0, t0

    def prepare_caches(self, prob):
        """
        Cache the fft objects, which are expensive to create on GPU because graphs have to be initialized.
        """
        prob.eval_f(prob.u_init)


class RBC3DAdaptivity(RayleighBenard3DRegular):
    Tend = 100

    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)

        # desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-4, 'dt_rel_min_slope': 0.1, 'dt_min': 1e-2}
        desc['level_params']['restol'] = -1
        desc['level_params']['dt'] = 2e-2
        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['step_params']['maxiter'] = 4
        desc['problem_params']['Rayleigh'] = 1e6
        return desc


class RBC3DBenchmarkRK(RayleighBenard3DRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3

        desc = super().get_description(*args, **kwargs)

        desc['sweeper_class'] = ARK3

        desc['level_params']['dt'] = 1e-3 / 5
        desc['level_params']['restol'] = -1

        desc['sweeper_params'] = {}

        desc['problem_params']['Rayleigh'] = 1e8

        desc['step_params']['maxiter'] = 1
        return desc


class RBC3DBenchmarkSDC(RayleighBenard3DRegular):
    Tend = 200

    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)

        desc['level_params']['dt'] = 5e-3
        desc['level_params']['restol'] = -1
        desc['level_params']['nsweeps'] = 4

        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-FLEX'
        desc['sweeper_params']['QE'] = 'PIC'

        desc['problem_params']['Rayleigh'] = 1e8
        desc['problem_params']['max_cached_factorizations'] = 16

        desc['step_params']['maxiter'] = 1
        return desc


class RBC3DscalingOld(RayleighBenard3DRegular):
    Tend = 21e-2

    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)

        desc['level_params']['dt'] = self.Tend / 21
        desc['level_params']['restol'] = -1
        desc['level_params']['nsweeps'] = 4

        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-FLEX'
        desc['sweeper_params']['QE'] = 'PIC'
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')

        desc['problem_params']['Rayleigh'] = 1e8
        desc['problem_params']['max_cached_factorizations'] = 16

        desc['step_params']['maxiter'] = 1
        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.hooks.log_work import LogWork

        params = super().get_controller_params(*args, **kwargs)
        params['hook_class'] = [LogWork]
        return params


class RBC3Dscaling(RayleighBenard3DRegular):
    Tend = 13e-2

    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)

        desc['level_params']['dt'] = 1e-2
        desc['level_params']['restol'] = -1
        desc['level_params']['nsweeps'] = 4

        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['QE'] = 'PIC'
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')

        desc['problem_params']['Rayleigh'] = 1e8
        desc['problem_params']['max_cached_factorizations'] = 4

        desc['step_params']['maxiter'] = 1
        return desc

    def get_controller_params(self, *args, **kwargs):
        from pySDC.implementations.hooks.log_work import LogWork

        params = super().get_controller_params(*args, **kwargs)
        params['hook_class'] = [LogWork]
        return params


class RBC3DscalingIterative(RBC3Dscaling):
    ic_res = 128
    ic_time = 10.354437173596336
    Tend = 9e-2 + 10.354437173596336

    def get_description(self, *args, **kwargs):
        desc = super().get_description(*args, **kwargs)
        desc['level_params']['dt'] = 1e-2
        # desc['level_params']['nsweeps'] = 1
        # desc['level_params']['restol'] = 1e-5
        # desc['level_params']['e_tol'] = 1e-5
        # desc['step_params']['maxiter'] = 99
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['problem_params']['solver_type'] = 'gmres+ilu'  #'bicgstab+ilu'
        desc['problem_params']['solver_args'] = {'atol': 1e-8, 'rtol': 1e-8, 'maxiter': 1000}
        desc['problem_params']['preconditioner_args'] = {'fill_factor': 5, 'drop_tol': 1e-2}
        desc['sweeper_params']['skip_residual_computation'] = ()
        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        from pySDC.helpers.fieldsIO import Rectilinear

        _P = type(P)(nx=self.ic_res, ny=self.ic_res, nz=self.ic_res, comm=P.comm, useGPU=P.useGPU)
        _P.setUpFieldsIO()

        ic_path = f'{self.base_path}/data/{type(self).__name__}-res{self.ic_res}-ic.pySDC'
        try:
            ic_file = Rectilinear.fromFile(ic_path)
        except FileNotFoundError as err:
            if self.args['res'] == self.ic_res:
                from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
                from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter

                _args = self.args
                _args['o'] = ic_path
                _args['logger_level'] = 15

                def _get_LogToFile(self, *args, **kwargs):
                    if self.comms[1].rank > 0:
                        return None
                    import numpy as np
                    from pySDC.implementations.hooks.log_solution import LogToFile

                    LogToFile.filename = ic_path
                    LogToFile.time_increment = 1e-1
                    LogToFile.allow_overwriting = True

                    return LogToFile

                RBC3DAdaptivity.get_LogToFile = _get_LogToFile

                config = RBC3DAdaptivity(args=self.args, comm_world=self.comm_world)

                description = config.get_description(
                    useGPU=_args['useGPU'], MPIsweeper=_args['procs'][1] > 1, res=_args['res']
                )
                description['problem_params']['max_cached_factorizations'] = description['sweeper_params']['num_nodes']
                description['convergence_controllers'][StepSizeLimiter] = {'dt_min': 1e-3}
                controller_params = config.get_controller_params(logger_level=_args['logger_level'])

                assert config.comms[0].size == 1
                controller = controller_nonMPI(
                    num_procs=1, controller_params=controller_params, description=description
                )
                prob = controller.MS[0].levels[0].prob

                u0, t0 = config.get_initial_condition(prob, restart_idx=_args['restart_idx'])

                config.prepare_caches(prob)

                uend, stats = controller.run(u0=u0, t0=t0, Tend=self.ic_time)

                ic_file = Rectilinear.fromFile(ic_path)
                assert (
                    self.ic_time in ic_file.times
                ), f'IC time {self.ic_time} not in recorded times! Got only {ic_file.times}'
            else:
                raise FileNotFoundError(
                    f'No ICs found for this configuration at path {ic_path!r}! Please run once with resolution {self.ic_res} to generate ICs.'
                ) from err

        # interpolate the solution using padded transforms
        padding_factor = self.ic_res / self.args['res']

        ic_idx = ic_file.times.index(self.ic_time)
        _, ics = ic_file.readField(ic_idx)

        ics = _P.xp.array(ics)
        _ics_hat = _P.transform(ics)
        ics_large = _P.itransform(_ics_hat, padding=(1 / padding_factor,) * (ics.ndim - 1))

        P.setUpFieldsIO()
        if P.spectral_space:
            u0_hat = P.u_init_forward
            u0_hat[...] = P.transform(ics_large)
            return u0_hat, self.ic_time
        else:
            return ics_large, self.ic_time


class RBC3Dverification(RayleighBenard3DRegular):
    converged = 0
    dt = 1e-2
    ic_config = None
    res = None
    Ra = None
    Tend = 100

    def get_file_name(self):
        res = self.args['res']
        dt = self.args['dt']
        return f'{self.base_path}/data/{type(self).__name__}-res{res}-dt{dt:.0e}.pySDC'

    def get_description(self, *args, res=-1, dt=-1, **kwargs):
        desc = super().get_description(*args, **kwargs)
        desc['level_params']['nsweeps'] = 4
        desc['level_params']['restol'] = -1
        desc['step_params']['maxiter'] = 1
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params']['num_nodes'] = 4
        Ra = int(type(self).__name__[-3]) * 10 ** int(type(self).__name__[-1])
        desc['problem_params']['Rayleigh'] = Ra

        _res = self.res if res == -1 else res
        desc['problem_params']['nx'] = _res
        desc['problem_params']['ny'] = _res
        desc['problem_params']['nz'] = _res

        _dt = self.dt if dt == -1 else dt
        desc['level_params']['dt'] = _dt
        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):
        if self.ic_config is None or restart_idx > 0:
            return super().get_initial_condition(P, *args, restart_idx=restart_idx, **kwargs)

        # read initial conditions
        from pySDC.helpers.fieldsIO import FieldsIO

        ic_config = self.ic_config(args={**self.args, 'res': -1, 'dt': -1})
        desc = ic_config.get_description()
        ic_res = desc['problem_params']['nz']

        _P = type(P)(nx=ic_res, ny=ic_res, nz=ic_res, comm=P.comm, useGPU=P.useGPU)
        _P.setUpFieldsIO()
        filename = ic_config.get_file_name()
        ic_file = FieldsIO.fromFile(filename)
        t0, ics = ic_file.readField(-1)
        P.logger.info(f'Loaded initial conditions from {filename!r} at t={t0}.')

        # interpolate the initial conditions using padded transforms
        padding_factor = ic_res / P.nz

        ics = _P.xp.array(ics)
        _ics_hat = _P.transform(ics)
        ics_interpolated = _P.itransform(_ics_hat, padding=(1 / padding_factor,) * (ics.ndim - 1))

        self.get_LogToFile()

        P.setUpFieldsIO()
        if P.spectral_space:
            u0_hat = P.u_init_forward
            u0_hat[...] = P.transform(ics_interpolated)
            return u0_hat, 0
        else:
            return ics_interpolated, 0


class RBC3DverificationRK(RBC3Dverification):

    def get_description(self, *args, res=-1, dt=-1, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3

        desc = super().get_description(*args, res=res, dt=dt, **kwargs)
        desc['level_params']['nsweeps'] = 1
        desc['level_params']['restol'] = -1
        desc['step_params']['maxiter'] = 1
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params'].pop('QI')
        desc['sweeper_params'].pop('num_nodes')
        desc['sweeper_class'] = ARK3
        return desc


class RBC3DRa1e4(RBC3Dverification):
    # converged = 60
    dt = 1.0
    ic_config = None
    res = 32


class RBC3DRKRa1e4(RBC3DverificationRK):
    # converged = 60
    dt = 1.0
    ic_config = None
    res = 32


class RBC3DRa1e5(RBC3Dverification):
    # converged = 40
    dt = 1e-1
    ic_config = RBC3DRa1e4
    res = 32


class RBC3DRKRa1e5(RBC3DverificationRK):
    dt = 1e-1
    ic_config = RBC3DRa1e4
    res = 32


class RBC3DRa1e6(RBC3Dverification):
    dt = 1e-1
    ic_config = RBC3DRa1e5
    res = 32


class RBC3DRa1e7(RBC3Dverification):
    dt = 5e-2
    ic_config = RBC3DRa1e6
    res = 64
