import numpy as np
from matplotlib.colors import TABLEAU_COLORS
cmap = TABLEAU_COLORS

class Strategy:
    '''
    Abstract class for resilience strategies
    '''

    def __init__(self):
        '''
        Initialization routine
        '''

        # set default values for plotting
        self.linestyle = '-'
        self.marker = '.'
        self.name = ''
        self.bar_plot_x_label = ''
        self.color = list(cmap.values())[0]

        # setup custom descriptions
        self.custom_description = {}

        # prepare parameters for masks to identify faults that cannot be fixed by this strategy
        self.fixable = []
        self.fixable += [
            {
                'key': 'node',
                'op': 'gt',
                'val': 0,
            }
        ]
        self.fixable += [
            {
                'key': 'error',
                'op': 'isfinite',
            }
        ]

    def get_fixable_params(self, **kwargs):
        """
        Return a list containing dictionaries which can be passed to `FaultStats.get_mask` as keyword arguments to
        obtain a mask of faults that can be fixed

        Returns:
            list: Dictionary of parameters
        """
        return self.fixable

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that realizes the resilience strategy and tailors it to the problem at hand

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: The custom descriptions you can supply to the problem when running it
        '''

        return self.custom_description

    def get_fault_args(self, problem, num_procs):
        '''
        Routine to get arguments for the faults that are exempt from randomization

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Arguments for the faults that are exempt from randomization
        '''

        return {}

    def get_random_params(self, problem, num_procs):
        '''
        Routine to get parameters for the randomization of faults

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            dict: Randomization parameters
        '''

        return {}

    @property
    def style(self):
        """
        Get the plotting parameters for the strategy.
        Supply them to a plotting function using `**`

        Returns:
            (dict): The plotting parameters as a dictionary
        """
        return {
            'marker': self.marker,
            'label': self.label,
            'color': self.color,
            'ls': self.linestyle,
        }

    @property
    def label(self):
        """
        Get a label for plotting
        """
        return self.name


class BaseStrategy(Strategy):
    '''
    Do a fixed iteration count
    '''

    def __init__(self):
        '''
        Initialization routine
        '''
        super(BaseStrategy, self).__init__()
        self.color = list(cmap.values())[0]
        self.marker = 'o'
        self.name = 'base'
        self.bar_plot_x_label = 'base'

    @property
    def label(self):
        return r'fixed'


class AdaptivityStrategy(Strategy):
    '''
    Adaptivity as a resilience strategy
    '''

    def __init__(self):
        '''
        Initialization routine
        '''
        super(AdaptivityStrategy, self).__init__()
        self.color = list(cmap.values())[1]
        self.marker = '*'
        self.name = 'adaptivity'
        self.bar_plot_x_label = 'adaptivity'

    def get_fixable_params(self, maxiter, **kwargs):
        """
        Here faults occurring in the last iteration cannot be fixed.

        Args:
            maxiter (int): Max. iterations until convergence is declared

        Returns:
            (list): Contains dictionaries of keyword arguments for `FaultStats.get_mask`
        """
        self.fixable += [
            {
                'key': 'iteration',
                'op': 'lt',
                'val': maxiter,
            }
        ]
        return self.fixable

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom descriptions you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
         
        custom_description = {}

        dt_max = np.inf
        dt_min = 1e-5

        if problem.__name__ == "run_piline":
            e_tol = 1e-7
            dt_min = 1e-2
        elif problem.__name__ == "run_vdp":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Lorenz":
            e_tol = 2e-5
            dt_min = 1e-3
        elif problem.__name__ == "run_Schroedinger":
            e_tol = 4e-6
            dt_min = 1e-3
        elif problem.__name__ == "run_leaky_superconductor":
            e_tol = 1e-7
            dt_min = 1e-3
            dt_max = 1e2
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptivity for your problem. Please add one to the\
 strategy'
            )

        custom_description['convergence_controllers'] = {
            Adaptivity: {'e_tol': e_tol, 'dt_min': dt_min, 'dt_max': dt_max}
        }
        return {**custom_description, **self.custom_description}


class AdaptiveHotRodStrategy(Strategy):
    '''
    Adaptivity + Hot Rod as a resilience strategy
    '''

    def __init__(self):
        '''
        Initialization routine
        '''
        super(AdaptiveHotRodStrategy, self).__init__()
        self.color = list(cmap.values())[4]
        self.marker = '.'
        self.name = 'adaptive Hot Rod'
        self.bar_plot_x_label = 'adaptive\nHot Rod'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds adaptivity and Hot Rod

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
         
        if problem.__name__ == "run_vdp":
            e_tol = 3e-7
            dt_min = 1e-3
            maxiter = 4
            HotRod_tol = 3e-7
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for adaptive Hot Rod for your problem. Please add one \
to the strategy'
            )

        no_storage = num_procs > 1

        custom_description = {
            'convergence_controllers': {
                Adaptivity: {'e_tol': e_tol, 'dt_min': dt_min},
                HotRod: {'HotRod_tol': HotRod_tol, 'no_storage': no_storage},
            },
            'step_params': {'maxiter': maxiter},
        }

        return {**custom_description, **self.custom_description}


class IterateStrategy(Strategy):
    '''
    Iterate for as much as you want
    '''

    def __init__(self):
        '''
        Initialization routine
        '''
        super(IterateStrategy, self).__init__()
        self.color = list(cmap.values())[2]
        self.marker = 'v'
        self.name = 'iterate'
        self.bar_plot_x_label = 'iterate'

    @property
    def label(self):
        return r'$k$ adaptivity'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that allows for adaptive iteration counts

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        restol = -1
        e_tol = -1

        if problem.__name__ == "run_piline":
            restol = 2.3e-8
        elif problem.__name__ == "run_vdp":
            restol = 9e-7
        elif problem.__name__ == "run_Lorenz":
            restol = 16e-7
        elif problem.__name__ == "run_Schroedinger":
            restol = 6.5e-7
        elif problem.__name__ == "run_leaky_superconductor":
            # e_tol = 1e-6
            restol = 1e-11
        else:
            raise NotImplementedError(
                'I don\'t have a residual tolerance for your problem. Please add one to the \
strategy'
            )

        custom_description = {
            'step_params': {'maxiter': 99},
            'level_params': {'restol': restol, 'e_tol': e_tol},
        }

        if problem.__name__ == "run_leaky_superconductor":
            custom_description['level_params']['dt'] = 26

        return {**custom_description, **self.custom_description}


class HotRodStrategy(Strategy):
    '''
    Hot Rod as a resilience strategy
    '''

    def __init__(self):
        '''
        Initialization routine
        '''
        super(HotRodStrategy, self).__init__()
        self.color = list(cmap.values())[3]
        self.marker = '^'
        self.name = 'Hot Rod'
        self.bar_plot_x_label = 'Hot Rod'

    def get_custom_description(self, problem, num_procs):
        '''
        Routine to get a custom description that adds Hot Rod

        Args:
            problem: A function that runs a pySDC problem, see imports for available problems
            num_procs (int): Number of processes you intend to run with

        Returns:
            The custom description you can supply to the problem when running it
        '''
        from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
         
        if problem.__name__ == "run_vdp":
            HotRod_tol = 5e-7
            maxiter = 4
        elif problem.__name__ == "run_Lorenz":
            HotRod_tol = 4e-7
            maxiter = 6
        elif problem.__name__ == "run_Schroedinger":
            HotRod_tol = 3e-7
            maxiter = 6
        elif problem.__name__ == "run_leaky_superconductor":
            HotRod_tol = 3e-5
            maxiter = 6
        else:
            raise NotImplementedError(
                'I don\'t have a tolerance for Hot Rod for your problem. Please add one to the\
 strategy'
            )

        no_storage = num_procs > 1

        custom_description = {
            'convergence_controllers': {
                HotRod: {'HotRod_tol': HotRod_tol, 'no_storage': no_storage},
                BasicRestartingNonMPI: {'max_restarts': 2, 'crash_after_max_restarts': False},
            },
            'step_params': {'maxiter': maxiter},
        }

        return {**custom_description, **self.custom_description}
