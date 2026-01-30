import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_config(fname, ax):  # pragma: no cover
    data = pd.read_csv(fname)

    for _res in np.unique(data.res):
        res_mask = data.res == _res

        for _tasks_time in np.unique(data.ntasks_time):
            task_mask = data.ntasks_time == _tasks_time
            Pint_label = " PinT" if _tasks_time > 1 else ""

            for _distribution in np.unique(data.distribution):
                distribution_mask = data.distribution == _distribution

                mask = np.logical_and(np.logical_and(res_mask, task_mask), distribution_mask)
                procs = np.array(data.procs[mask])
                time = np.array(data.time[mask])
                ax.loglog(
                    procs[np.argsort(procs)],
                    time[np.argsort(procs)],
                    label=fr'$N_z$={_res}{Pint_label} {_distribution}',
                )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('tasks')
    ax.set_ylabel('time / s')


def plot_RBC3DG4R4SDC23Ra1e5():  # pragma: no cover
    fig, ax = plt.subplots()
    fname = './results/JUSUF_RBC3DG4R4SDC23Ra1e5.txt'
    plot_config(fname, ax)
    ax.legend(frameon=False)


def plot_RBC3DG4R4SDC44Ra1e5():  # pragma: no cover
    fig, ax = plt.subplots()
    fname = './results/JUSUF_RBC3DG4R4SDC44Ra1e5.txt'
    plot_config(fname, ax)
    ax.legend(frameon=False)


if __name__ == '__main__':
    plot_RBC3DG4R4SDC44Ra1e5()
    plt.show()
