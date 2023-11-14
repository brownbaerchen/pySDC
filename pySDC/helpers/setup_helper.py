def generate_description(problem_class, **kwargs):
    """
    Generate a description object that you can use to run pySDC based on a problem class and various input.
    This function does not set any additional defaults, but distributes the values to where they belong.

    Args:
        problem_class (pySDC.Problem): A problem class

    Returns:
        dict: A description object for running pySDC
    """
    from pySDC.core.Level import _Pars as level_params
    from pySDC.core.Step import _Pars as step_params

    description = {
        'level_params': {},
        'problem_params': {},
        'sweeper_params': {},
        'problem_class': problem_class,
        'step_params': {},
        'sweeper_class': kwargs.get('sweeper_class', problem_class.get_default_sweeper_class()),
        'convergence_controllers': {},
    }

    problem_keys = problem_class.__init__.__code__.co_varnames
    level_keys = level_params({}).__dict__.keys()
    sweeper_keys = description['sweeper_class']({'num_nodes': 1, 'quad_type': 'RADAU-RIGHT'}).params.__dict__.keys()
    step_keys = step_params({}).__dict__.keys()

    # TODO: add convergence controllers
    for key, val in kwargs.items():
        if key in problem_keys:
            description['problem_params'][key] = val
        elif key in level_keys:
            description['level_params'][key] = val
        elif key in sweeper_keys:
            description['sweeper_params'][key] = val
        elif key in step_keys:
            description['step_params'][key] = val
        elif key == 'sweeper_class':
            pass
        else:
            raise ValueError(f'Don\'t know what parameter \"{key}\" is for!')

    return description


def get_xpy_module(description):
    import logging
    logger = logging.getLogger('setup helper')

    dtype = description['problem_class'].dtype_u.__name__
    use_cupy = description['problem_params'].get('useGPU', False)

    if dtype in ['cupy_mesh'] or use_cupy:
        logger.debug('Selected cupy module in order to run on GPU')
        import cupy
        return cupy
    if dtype in ['mesh']:
        logger.debug(f'Selected numpy module in order to run on CPU based on \"{dtype}\" datatype')
        import numpy
        return numpy
    else:
        logger.warning(f'Don\'t know whether to use numpy or cupy, defaulting to numpy')
        import numpy
        return numpy

