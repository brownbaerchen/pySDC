import pytest


def add_step(hook, time, step_size, restart, value, **kwargs):
    base_values = {
        'process': 0,
        'level': 0,
        'iter': 0,
        'sweep': 0,
        **kwargs,
    }

    hook.add_to_stats(**base_values, time=time, value=value, type='beginning')
    hook.add_to_stats(**base_values, time=time + step_size, value=value, type='end')

    if restart:
        hook._hooks__num_restarts += 1

    for t in [time, time + step_size]:
        hook.add_to_stats(process=-1, time=t, level=-1, iter=-1, sweep=-1, type='_recomputed', value=restart)


def generate_stats_for_recomputed_test_1(num_procs=1):
    """
    This function will add two values, one of which will be the updated version after a restart. We enter the value True for the values that are supposed to remain after filtering and we add False for values we want removed by filtering.
    """
    from pySDC.core.Hooks import hooks

    hook = hooks()

    step_size = 1.0

    # first step
    for rank in range(num_procs):
        add_step(hook, time=0 + step_size * rank, step_size=1.0, restart=True, value=False, process=rank)

    # repeat first step
    for rank in range(num_procs):
        add_step(hook, time=0 + step_size * rank, step_size=1.0, restart=False, value=True, process=rank)

    # second step
    for rank in range(num_procs):
        add_step(
            hook,
            time=0 + step_size * rank + step_size * num_procs,
            step_size=1.0,
            restart=False,
            value=True,
            process=rank,
        )

    return hook.return_stats()


def generate_stats_for_recomputed_test_2(num_procs=1):
    """
    This function will add two values, one of which will be the updated version after a restart. We enter the value True for the values that are supposed to remain after filtering and we add False for values we want removed by filtering.
    """
    from pySDC.core.Hooks import hooks

    hook = hooks()

    step_size = 1.0

    # first step
    for rank in range(num_procs):
        add_step(hook, time=0 + step_size * rank, step_size=step_size, restart=True, value=False, process=rank)

    step_size = 0.1

    # repeat first step
    for rank in range(num_procs):
        add_step(hook, time=0 + step_size * rank, step_size=step_size, restart=False, value=True, process=rank)

    # second step
    for rank in range(num_procs):
        add_step(
            hook,
            time=step_size * num_procs + step_size * rank,
            step_size=step_size,
            restart=False,
            value=True,
            process=rank,
        )

    return hook.return_stats()


@pytest.mark.base
@pytest.mark.parametrize("test_type", [1, 2])
@pytest.mark.parametrize("num_procs", [1, 4])
def test_filter_recomputed(test_type, num_procs):
    from pySDC.helpers.stats_helper import filter_stats

    if test_type == 1:
        stats = generate_stats_for_recomputed_test_1(num_procs=num_procs)
        msg = "Error when filtering values that have been superseded after a restart!"
    elif test_type == 2:
        stats = generate_stats_for_recomputed_test_2(num_procs=num_procs)
        msg = "Error when filtering values that are not superseded after a  restart!"
    else:
        raise NotImplementedError(f'Test type {test_type} not implemented')

    # extract only the variables we want to examine and get rid of all "helper" variables in the stats
    filtered = {**filter_stats(stats.copy(), type='beginning'), **filter_stats(stats.copy(), type='end')}
    filtered_recomputed = {
        **filter_stats(stats.copy(), type='beginning', recomputed=False),
        **filter_stats(stats.copy(), type='end', recomputed=False),
    }

    assert all(
        val for key, val in filtered_recomputed.items()
    ), f'{msg} Some unwanted values have made it into the filtered stats!\nFull stats: {filtered},\nFiltered: {filtered_recomputed}'
    assert len(filtered_recomputed) == len(
        [val for key, val in filtered.items() if val]
    ), f'{msg} Too many values have been removed in the process of filtering!\nFull stats: {filtered},\nFiltered: {filtered_recomputed}'
    assert len(filtered_recomputed) < len(
        stats
    ), f'{msg} Apparently, nothing was filtered!\nFull stats: {filtered},\nFiltered: {filtered_recomputed}'


if __name__ == '__main__':
    test_filter_recomputed(1)
