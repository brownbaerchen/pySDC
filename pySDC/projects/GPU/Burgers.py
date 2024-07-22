from mpi4py import MPI
import numpy as np
from pySDC.implementations.hooks.log_solution import LogToFile

# class LogVorticity(LogToFile):


def run_Burgers():
    from pySDC.implementations.problem_classes.Burgers import Burgers2D
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    # from pySDC.implementations.
    from pySDC.implementations.hooks.live_plotting import PlotPostStep
    from pySDC.projects.GPU.hooks.LogGrid import LogGrid

    comm = MPI.COMM_WORLD
    LogToFile.path = './data/'
    LogToFile.file_name = f'Burgers-{comm.rank}'
    LogToFile.process_solution = lambda L: {
        't': L.time + L.dt,
        'u': L.uend.view(np.ndarray),
        'vorticity': L.prob.compute_vorticity(L.uend).view(np.ndarray),
    }
    LogGrid.file_name = f'Burgers-grid-{comm.rank}'
    LogGrid.file_logger = LogToFile

    level_params = {}
    level_params['dt'] = 1e-2

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 1
    sweeper_params['QI'] = 'IE'
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'comm': comm,
        'nx': 2**8,
        'nz': 2**8,
        'epsilon': 1e-1,
    }

    step_params = {}
    step_params['maxiter'] = 1

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogToFile, LogGrid]
    controller_params['mssdc_jac'] = False

    description = {}
    description['problem_class'] = Burgers2D
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 0.0
    Tend = 4
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


def plot_Burgers(size):
    import matplotlib.pyplot as plt
    from pySDC.implementations.hooks.log_solution import LogToFile
    from pySDC.projects.GPU.hooks.LogGrid import LogGrid
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    LogToFile.path = './data/'
    LogGrid.file_logger = LogToFile

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.03)

    for i in range(400):
        buffer = {}
        vmin = 0
        vmax = 0
        for rank in range(size):
            LogToFile.file_name = f'Burgers-{rank}'
            LogGrid.file_name = f'Burgers-grid-{rank}'

            buffer[f'u-{rank}'] = LogToFile.load(i)
            buffer[f'Z-{rank}'], buffer[f'X-{rank}'] = LogGrid.load()

            vmin = min([vmin, buffer[f'u-{rank}']['vorticity'].real.min()])
            vmax = max([vmax, buffer[f'u-{rank}']['vorticity'].real.max()])

        for rank in range(size):
            im = ax.pcolormesh(
                buffer[f'X-{rank}'], buffer[f'Z-{rank}'], buffer[f'u-{rank}']['vorticity'].real, vmin=vmin, vmax=vmax
            )
            ax.set_title(f't={buffer[f"u-{rank}"]["t"]:.2e}')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            fig.colorbar(im, cax)
        plt.pause(1e-9)
        fig.savefig(f'simulation_plots/Burgers_{i:06d}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # run_Burgers()
    if MPI.COMM_WORLD.rank == 0:
        plot_Burgers(MPI.COMM_WORLD.size)
