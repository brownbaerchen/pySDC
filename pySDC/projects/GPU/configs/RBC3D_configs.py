from pySDC.projects.GPU.configs.base_config import Config


def get_config(args):
    name = args['config']
    if name == 'RBC3D':
        return RayleighBenard3DRegular(args)
    elif name == 'RBC3DAdaptivity':
        return RBC3DAdaptivity(args)
    elif name == 'RBC3DBenchmarkRK':
        return RBC3DBenchmarkRK(args)
    elif name == 'RBC3DBenchmarkSDC':
        return RBC3DBenchmarkSDC(args)
    elif name == 'RBC3Dscaling':
        return RBC3Dscaling(args)
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
        LogToFile.time_increment = 1e-1
        # LogToFile.allow_overwriting = True

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

        desc['step_params']['maxiter'] = 3

        desc['problem_class'] = RayleighBenard3D

        return desc

    def get_initial_condition(self, P, *args, restart_idx=0, **kwargs):

        if restart_idx == 0:
            u0 = P.u_exact(t=0, seed=P.comm.rank, noise_level=1e-3)
            u0_with_pressure = P.solve_system(u0, 1e-9, u0)
            return u0_with_pressure, 0
        else:
            from pySDC.helpers.fieldsIO import FieldsIO

            P.setUpFieldsIO()
            outfile = FieldsIO.fromFile(self.get_file_name())

            t0, solution = outfile.readField(restart_idx)

            u0 = P.u_init
            u0[...] = solution[:]

            if P.spectral_space:
                u0[...] = P.transform(u0)
            return u0, t0

    def prepare_caches(self, prob):
        """
        Cache the fft objects, which are expensive to create on GPU because graphs have to be initialized.
        """
        prob.eval_f(prob.u_init)


class RBC3DAdaptivity(RayleighBenard3DRegular):
    def get_description(self, *args, **kwargs):
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        desc = super().get_description(*args, **kwargs)

        desc['convergence_controllers'][Adaptivity] = {'e_tol': 1e-4, 'dt_rel_min_slope': 0.1}
        desc['level_params']['restol'] = -1
        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['skip_residual_computation'] = ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
        desc['sweeper_params']['QI'] = 'MIN-SR-S'
        desc['step_params']['maxiter'] = 4
        return desc


class RBC3DBenchmarkRK(RayleighBenard3DRegular):
    def get_description(self, *args, res=-1, **kwargs):
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK3

        desc = super().get_description(*args, **kwargs)

        desc['sweeper_class'] = ARK3

        desc['level_params']['dt'] = 1e-3 / 5
        desc['level_params']['restol'] = -1

        desc['sweeper_params'] = {}

        desc['problem_params']['Rayleigh'] = 1e8
        desc['problem_params']['nx'] = 64 if res == -1 else res
        desc['problem_params']['ny'] = desc['problem_params']['nx']
        desc['problem_params']['nz'] = desc['problem_params']['nx']

        desc['step_params']['maxiter'] = 1
        return desc


class RBC3DBenchmarkSDC(RayleighBenard3DRegular):
    def get_description(self, *args, res=-1, **kwargs):
        desc = super().get_description(*args, **kwargs)

        desc['level_params']['dt'] = 5e-3
        desc['level_params']['restol'] = -1
        desc['level_params']['nsweeps'] = 4

        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-FLEX'
        desc['sweeper_params']['QE'] = 'PIC'

        desc['problem_params']['Rayleigh'] = 1e8
        desc['problem_params']['nx'] = 64 if res == -1 else res
        desc['problem_params']['ny'] = desc['problem_params']['nx']
        desc['problem_params']['nz'] = desc['problem_params']['nx']
        desc['problem_params']['max_cached_factorizations'] = 16

        desc['step_params']['maxiter'] = 1
        return desc


class RBC3Dscaling(RayleighBenard3DRegular):
    Tend = 21e-2

    def get_description(self, *args, res=-1, **kwargs):
        desc = super().get_description(*args, **kwargs)

        desc['level_params']['dt'] = self.Tend / 21
        desc['level_params']['restol'] = -1
        desc['level_params']['nsweeps'] = 4

        desc['sweeper_params']['num_nodes'] = 4
        desc['sweeper_params']['QI'] = 'MIN-SR-FLEX'
        desc['sweeper_params']['QE'] = 'PIC'

        desc['problem_params']['Rayleigh'] = 1e8
        desc['problem_params']['nx'] = 64 if res == -1 else res
        desc['problem_params']['ny'] = desc['problem_params']['nx']
        desc['problem_params']['nz'] = desc['problem_params']['nx']
        desc['problem_params']['max_cached_factorizations'] = 16

        desc['step_params']['maxiter'] = 1
        return desc
