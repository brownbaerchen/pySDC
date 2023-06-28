import pytest


def add_step(hook, time, step_size, restart, value):
    base_values = {
        'process': 0,
        'level': 0,
        'iter': 0,
        'sweep': 0,
    }

    hook.add_to_stats(**base_values, time=time, value=value, type='beginning')
    hook.add_to_stats(**base_values, time=time + step_size, value=value, type='end')

    if restart:
        hook._hooks__num_restarts += 1

    for t in [time, time + step_size]:
        hook.add_to_stats(process=-1, time=t, level=-1, iter=-1, sweep=-1, type='_recomputed', value=restart)


def generate_stats_for_recomputed_test_1():
    """
    This function will add two values, one of which will be the updated version after a restart. We enter the value True for the values that are supposed to remain after filtering and we add False for values we want removed by filtering.
    """
    from pySDC.core.Hooks import hooks

    hook = hooks()

    # first step
    add_step(hook, time=0, step_size=1.0, restart=True, value=False)

    # repeat first step
    add_step(hook, time=0, step_size=1.0, restart=False, value=True)

    # second step
    add_step(hook, time=1, step_size=1.0, restart=False, value=True)

    return hook.return_stats()


def generate_stats_for_recomputed_test_2():
    """
    This function will add two values, one of which will be the updated version after a restart. We enter the value True for the values that are supposed to remain after filtering and we add False for values we want removed by filtering.
    """
    from pySDC.core.Hooks import hooks

    hook = hooks()

    # first step
    add_step(hook, time=0, step_size=1.0, restart=True, value=False)

    # repeat first step
    add_step(hook, time=0, step_size=0.1, restart=False, value=True)

    # second step
    add_step(hook, time=0.1, step_size=0.1, restart=False, value=True)

    return hook.return_stats()


@pytest.mark.base
@pytest.mark.parametrize("test_type", [1, 2])
def test_filter_recomputed(test_type):
    from pySDC.helpers.stats_helper import filter_recomputed, filter_stats

    if test_type == 1:
        stats = generate_stats_for_recomputed_test_1()
        msg = "Error when filtering values that have been superseded after a restart!"
    elif test_type == 2:
        stats = generate_stats_for_recomputed_test_2()
        msg = "Error when filtering values that are not superseded after a  restart!"
    else:
        raise NotImplementedError(f'Test type {test_type} not implemented')

    # filter the recomputed values out of all stats
    filter_recomputed_ = filter_recomputed(stats.copy())

    # extract only the variables we want to examine and get rid of all "helper" variables in the stats
    filtered = {**filter_stats(stats.copy(), type='beginning'), **filter_stats(stats.copy(), type='end')}
    filtered_recomputed = {
        **filter_stats(filter_recomputed_.copy(), type='beginning'),
        **filter_stats(filter_recomputed_.copy(), type='end'),
    }

    assert all(
        val for key, val in filtered_recomputed.items()
    ), f'{msg} Some unwanted values have made it into the filtered stats!\nFull stats: {filtered},\nFiltered: {filtered_recomputed}'
    assert len(filtered_recomputed) == len(
        [val for key, val in filtered.items() if val]
    ), f'{msg} Too many values have been removed in the process of filtering!'
    assert len(filtered_recomputed) < len(stats), f'{msg} Apparently, nothing was filtered!'


if __name__ == '__main__':
    test_filter_recomputed(1)
