import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.Resilience.strategies import BaseStrategy
from pySDC.projects.Resilience.advection import run_advection

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

from pySDC.implementations.convergence_controller_classes.compression import Compression

def single_run(problem, description=None):
    description = {} if description is None else description
    description['convergence_controllers'] = {Compression: {}}
    stats, _, _ = problem(custom_description=description, hook_class=LogGlobalErrorPostRun)
    e = get_sorted(stats, type='e_global_post_run')[-1][1]
    return e


def multiple_runs(problem, values, expected_order, mode='dt', **kwargs):
    errors = np.zeros_like(values)

    description = {
        'level_params': {},
        'problam_params': {},
        'step_params': {},
    }
    if mode == 'dt':
        description['step_params'] = {'maxiter': expected_order}
    elif mode == 'nvars':
        description['problem_params'] = {'order': expected_order}
         
    for i in range(len(values)):
        if mode == 'dt':
            description['level_params']['dt'] = values[i]
        elif mode == 'nvars':
            description['problem_params']['nvars'] = values[i]
             
        errors[i] = single_run(problem, description)
    print(errors)
    return values, errors


def get_order(values, erros, thresh=1e-16):
    values = np.array(values)
    idx = np.argsort(values)
    local_orders = np.log(errors[idx][1:] / errors[idx][:-1]) / np.log(values[idx][1:] / values[idx][:-1])
    return np.mean(local_orders[errors[idx][1:] > thresh])


def plot_order(values, errors, ax, thresh=1e-16, color='black', **kwargs):
    values = np.array(values)
    order = get_order(values, errors, thresh=thresh)
    ax.scatter(values, errors, color=color, **kwargs)
    ax.loglog(values, errors[0] * (values/values[0]) ** order, color=color, label=f'p={order:.2f}', **kwargs)



if __name__ == '__main__':
    fig, ax = plt.subplots()
    problem = run_advection

    base_configs_dt = {
        'values': np.array([2.**(-i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        'mode': 'dt',
        'ax': ax
    }

    configs_dt = {}
    # configs_dt[2] = {**base_configs_dt, 'color': 'black'}
    # configs_dt[3] = {**base_configs_dt, 'color': 'magenta'}
    configs_dt[4] = {**base_configs_dt, 'color': 'teal'}
    # configs_dt[5] = {**base_configs_dt, 'color': 'orange'}
    # configs_dt[6] = {**base_configs_dt, 'color': 'blue'}
     
    for key in configs_dt.keys():
        values, errors = multiple_runs(problem, expected_order=key, **configs_dt[key])
        plot_order(values, errors, ax=configs_dt[key]['ax'], thresh=1e-9, color=configs_dt[key]['color'])
    base_configs_dt['ax'].set_xlabel(r'$\Delta t$')
    base_configs_dt['ax'].set_ylabel('global error')

    #base_configs_nvars = {
    #    'values': [128, 256, 512, 1024],
    #    # 'values': np.array([2**(i) for i in [4, 5, 6, 7, 8, 9]]),
    #    'mode': 'nvars',
    #}

    #configs_nvars = {}
    #configs_nvars[2] = {**base_configs_nvars, 'color': 'black'}
    #configs_nvars[4] = {**base_configs_nvars, 'color': 'magenta'}

    #for key in configs_nvars.keys():
    #    values, errors = multiple_runs(problem, expected_order=key, **configs_nvars[key])
    #    plot_order(values, errors, axs[1], color=configs_nvars[key]['color'])

    ax.legend(frameon=False)
    fig.savefig('compression.pdf')
    plt.show()
