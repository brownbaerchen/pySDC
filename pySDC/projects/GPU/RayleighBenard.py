from mpi4py import MPI
import numpy as np
from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile


def run_RBC(useGPU=False):
    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard, CFLLimit, SpaceAdaptivity
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityPolynomialError
    from pySDC.implementations.convergence_controller_classes.crash import StopAtNan
    from pySDC.implementations.problem_classes.generic_spectral import compute_residual_DAE

    # from pySDC.implementations.sweeper_classes.Runge_Kutta import ARK548L2SA as sweeper_class
    # from pySDC.implementations.sweeper_classes.Runge_Kutta import IMEXEuler as sweeper_class

    from pySDC.projects.GPU.hooks.LogGrid import LogGrid

    sweeper_class.compute_residual = compute_residual_DAE

    comm = MPI.COMM_WORLD
    LogToFile.path = './data/'
    LogToFile.file_name = f'RBC-{comm.rank}'

    if useGPU:
        LogToFile.process_solution = lambda L: {
            't': L.time + L.dt,
            'u': L.uend.get().view(np.ndarray),
            'X': L.prob.X.get().view(np.ndarray),
            'Z': L.prob.Z.get().view(np.ndarray),
            'vorticity': L.prob.compute_vorticity(L.uend).get().view(np.ndarray),
            'divergence': L.uend.xp.log10(L.uend.xp.abs(L.prob.compute_constraint_violation(L.uend)['divergence']))
            .get()
            .view(np.ndarray),
        }
    else:
        LogToFile.process_solution = lambda L: {
            't': L.time + L.dt,
            'u': L.uend.view(np.ndarray),
            'X': L.prob.X.view(np.ndarray),
            'Z': L.prob.Z.view(np.ndarray),
            'vorticity': L.prob.compute_vorticity(L.uend).view(np.ndarray),
            'divergence': L.uend.xp.log10(
                L.uend.xp.abs(L.prob.compute_constraint_violation(L.uend)['divergence'])
            ).view(np.ndarray),
        }
    LogToFile.time_increment = 1e-1
    LogGrid.file_name = f'RBC-grid-{comm.rank}'
    LogGrid.file_logger = LogToFile

    level_params = {}
    level_params['dt'] = 0.2
    level_params['restol'] = 1e-7

    convergence_controllers = {
        # Adaptivity: {'e_tol': 1e-5, 'dt_max': level_params['dt'], 'max_restarts': 33},
        # AdaptivityPolynomialError: {
        #     'e_tol': 1e-3,
        #     'interpolate_between_restarts': False,
        #     'dt_max': level_params['dt'],
        #     # 'dt_slope_max': 2,
        # },
        CFLLimit: {'dt_max': 2e-1, 'dt_min': 1e-6, 'cfl': 0.4},
        StopAtNan: {'thresh': 1e6},
        SpaceAdaptivity: {'nx_max': 2**10, 'nz_max': 2**8},
    }

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'comm': comm,
        'useGPU': useGPU,
        'Rayleigh': 2e6 / 16,
        'nx': 2**8 + 0,
        'nz': 2**6 + 0,
        'nx_max': 2**10 + 0,
        'nz_max': 2**8 + 0,
        'cheby_mode': 'T2U',
        'dealiasing': 3 / 2,
        # 'debug':True,
        'left_preconditioner': not useGPU,
        # 'right_preconditioning': 'T2T',
    }

    step_params = {}
    step_params['maxiter'] = 9

    controller_params = {}
    controller_params['logger_level'] = 15 if comm.rank == 0 else 40
    controller_params['hook_class'] = [LogToFile, LogGrid]
    controller_params['mssdc_jac'] = False

    description = {}
    description['problem_class'] = RayleighBenard
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 0.0
    Tend = 1
    P = controller.MS[0].levels[0].prob

    relaxation_steps = 0
    u0_noise = P.u_exact(t0, seed=comm.rank, noise_level=1e-3)
    uinit = u0_noise
    for _ in range(relaxation_steps):
        uinit = P.solve_system(uinit, dt=0.25)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


def plot_RBC(size, quantitiy='T', quantitiy2='vorticity', render=True, start_idx=0):
    import matplotlib.pyplot as plt
    from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile
    from pySDC.projects.GPU.hooks.LogGrid import LogGrid
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import gc

    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenard

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    P = RayleighBenard()

    LogToFile.path = './data/'
    LogGrid.file_logger = LogToFile

    cmaps = {'vorticity': 'bwr', 'p': 'bwr'}

    for i in range(start_idx + comm.rank, 9999, comm.size):
        fig = P.get_fig()
        cax = P.cax
        axs = fig.get_axes()

        buffer = {}
        vmin = {quantitiy: np.inf, quantitiy2: np.inf}
        vmax = {quantitiy: -np.inf, quantitiy2: -np.inf}
        for rank in range(size):
            LogToFile.file_name = f'RBC-{rank}'
            LogGrid.file_name = f'RBC-grid-{rank}'

            buffer[f'u-{rank}'] = LogToFile.load(i)
            buffer[f'Z-{rank}'], buffer[f'X-{rank}'] = LogGrid.load()

            vmin[quantitiy2] = min([vmin[quantitiy2], buffer[f'u-{rank}'][quantitiy2].real.min()])
            vmax[quantitiy2] = max([vmax[quantitiy2], buffer[f'u-{rank}'][quantitiy2].real.max()])
            vmin[quantitiy] = min([vmin[quantitiy], buffer[f'u-{rank}']['u'][P.index(quantitiy)].real.min()])
            vmax[quantitiy] = max([vmax[quantitiy], buffer[f'u-{rank}']['u'][P.index(quantitiy)].real.max()])

        for rank in range(size):
            im = axs[1].pcolormesh(
                buffer[f'u-{rank}']['X'],
                buffer[f'u-{rank}']['Z'],
                buffer[f'u-{rank}'][quantitiy2].real,
                vmin=-vmax[quantitiy2] if cmaps.get(quantitiy2, None) in ['bwr'] else vmin[quantitiy2],
                vmax=vmax[quantitiy2],
                cmap=cmaps.get(quantitiy2, None),
            )
            fig.colorbar(im, cax[1])
            im = axs[0].pcolormesh(
                buffer[f'u-{rank}']['X'],
                buffer[f'u-{rank}']['Z'],
                buffer[f'u-{rank}']['u'][P.index(quantitiy)].real,
                # vmin=-vmax[quantitiy] if cmaps.get(quantitiy, None) in ['bwr', 'seismic'] else vmin[quantitiy],
                vmin=vmin[quantitiy],
                vmax=-vmin[quantitiy] if cmaps.get(quantitiy, None) in ['bwr', 'seismic'] else vmax[quantitiy],
                # vmax=vmax[quantitiy],
                cmap=cmaps.get(quantitiy, 'plasma'),
            )
            fig.colorbar(im, cax[0])
            axs[0].set_title(f't={buffer[f"u-{rank}"]["t"]:.2f}')
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('z')
            axs[0].set_aspect(1.0)
            axs[1].set_aspect(1.0)
        path = f'simulation_plots/RBC{i:06d}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'{comm.rank} Stored figure {path!r} at t={buffer[f"u-{rank}"]["t"]:.2f}', flush=True)
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
        run_RBC(useGPU=args.useGPU)
    if args.plot:
        plot_RBC(size=args.np, render=args.render, start_idx=args.startIdx)
