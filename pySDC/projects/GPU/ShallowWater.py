from mpi4py import MPI
import numpy as np
from pySDC.implementations.hooks.log_solution import LogToFile


def run_SWE():
    from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE
    from pySDC.implementations.problem_classes.ShallowWater import ShallowWaterLinearized
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityPolynomialError

    generic_implicit.compute_residual = compute_residual_DAE

    from pySDC.implementations.hooks.live_plotting import PlotPostStep
    from pySDC.projects.GPU.hooks.LogGrid import LogGrid

    comm = MPI.COMM_WORLD
    LogToFile.path = './data/'
    LogToFile.file_name = f'SWE-{comm.rank}'
    LogToFile.process_solution = lambda L: {
        't': L.time + L.dt,
        'u': L.uend.view(np.ndarray),
        # 'vorticity': L.prob.compute_vorticity(L.uend).view(np.ndarray),
    }
    LogGrid.file_name = f'SWE-grid-{comm.rank}'
    LogGrid.file_logger = LogToFile

    convergence_controllers = {
        AdaptivityPolynomialError: {
            'e_tol': 1.7e-3,
            'abort_at_growing_residual': False,
            'interpolate_between_restarts': False,
        },
        # Adaptivity: {'e_tol': 1e-1},
    }

    level_params = {}
    level_params['dt'] = 1e-2
    level_params['restol'] = 1e-8
    # level_params['e_tol'] = 1e-3

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'comm': comm,
        'nx': 2**8,
        'nz': 2**8,
        'epsilon': 1e-1,
    }

    step_params = {}
    step_params['maxiter'] = 16

    controller_params = {}
    controller_params['logger_level'] = 11 if comm.rank == 0 else 40
    controller_params['hook_class'] = [LogToFile, LogGrid]
    controller_params['mssdc_jac'] = False

    description = {}
    description['problem_class'] = ShallowWaterLinearized
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 0.0
    Tend = 4
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


def plot_SWE(size, quantitiy='h', render=True, start_idx=0):
    import matplotlib.pyplot as plt
    from pySDC.implementations.hooks.log_solution import LogToFile
    from pySDC.projects.GPU.hooks.LogGrid import LogGrid
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import gc

    from pySDC.implementations.problem_classes.ShallowWater import ShallowWaterLinearized

    P = ShallowWaterLinearized()

    LogToFile.path = './data/'
    LogGrid.file_logger = LogToFile

    cmaps = {'vorticity': 'bwr'}

    for i in range(start_idx, 999):
        fig = P.get_fig()
        cax = P.cax
        axs = fig.get_axes()

        buffer = {}
        vmin = {quantitiy: np.inf}
        vmax = {quantitiy: -np.inf}
        for rank in range(size):
            LogToFile.file_name = f'SWE-{rank}'
            LogGrid.file_name = f'SWE-grid-{rank}'

            buffer[f'u-{rank}'] = LogToFile.load(i)
            buffer[f'Z-{rank}'], buffer[f'X-{rank}'] = LogGrid.load()

            vmin[quantitiy] = min([vmin[quantitiy], buffer[f'u-{rank}']['u'][P.index(quantitiy)].real.min()])
            vmax[quantitiy] = max([vmax[quantitiy], buffer[f'u-{rank}']['u'][P.index(quantitiy)].real.max()])

        for rank in range(size):
            im = axs[0].pcolormesh(
                buffer[f'X-{rank}'],
                buffer[f'Z-{rank}'],
                buffer[f'u-{rank}']['u'][P.index(quantitiy)].real,
                vmin=vmin[quantitiy],
                vmax=vmax[quantitiy],
                # cmap='plasma',
            )
            axs[0].set_title(f't={buffer[f"u-{rank}"]["t"]:.2f}')
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('y')
            axs[0].set_aspect(1.0)
            fig.colorbar(im, cax[0])
            cax[0].set_label(quantitiy)
        path = f'simulation_plots/SWE{i:06d}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Stored figure {path!r}', flush=True)
        # plt.show()
        if render:
            plt.pause(1e-9)
        plt.close(fig)
        del fig
        del buffer
        gc.collect()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=bool)
    parser.add_argument('--plot', type=bool)
    parser.add_argument('--useGPU', type=bool)
    parser.add_argument('--render', type=bool)
    parser.add_argument('--startIdx', type=int, default=0)
    parser.add_argument('--np', type=int, default=1)

    args = parser.parse_args()

    if args.run:
        run_SWE()
    if MPI.COMM_WORLD.rank == 0 and args.plot:
        plot_SWE(size=args.np, render=args.render, start_idx=args.startIdx)
