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


class SingleGPUExperiment(Experiment):
    name = 'single_gpu'


class AdaptivityExperiment(Experiment):
    name = 'adaptivity'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prob.add_polynomial_adaptivity()


def get_problem(name):
    probs = {
        'Schroedinger': RunSchroedinger,
        'AC': RunAllenCahn,
        'Brusselator': RunBrusselator,
    }
    return probs[name]


def get_experiment(name):
    ex = {
        'singleGPU': SingleGPUExperiment,
        'adaptivity': AdaptivityExperiment,
    }
    return ex[name]
