from mpi4py import MPI
import numpy as np
from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile
import pickle
from pySDC.helpers.stats_helper import filter_stats, get_sorted
from pySDC.implementations.hooks.log_step_size import LogStepSize


def run_RBC(useGPU=False):
    from pySDC.implementations.problem_classes.RayleighBenard import (
        RayleighBenardUltraspherical,
        RayleighBenard,
        CFLLimit,
        SpaceAdaptivity,
        LogAnalysisVariables,
    )
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
            'divergence': L.uend.xp.log10(L.uend.xp.abs(L.prob.eval_f(L.uend).impl[L.prob.index('p')]))
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
            'divergence': L.uend.xp.log10(L.uend.xp.abs(L.prob.eval_f(L.uend).impl[L.prob.index('p')])),
            # 'divergence': L.uend.xp.log10(
            #     L.uend.xp.abs(L.prob.compute_constraint_violation(L.uend)['divergence'])
            # ).view(np.ndarray),
        }
    LogToFile.time_increment = 1e-1
    LogGrid.file_name = f'RBC-grid-{comm.rank}'
    LogGrid.file_logger = LogToFile

    level_params = {}
    level_params['dt'] = 0.1
    level_params['restol'] = -1e-7

    convergence_controllers = {
        Adaptivity: {'e_tol': 1e-6, 'dt_max': level_params['dt']},
        # AdaptivityPolynomialError: {
        #     'e_tol': 1e-3,
        #     'interpolate_between_restarts': False,
        #     # 'dt_max': level_params['dt'],
        #     # 'dt_slope_max': 2,
        # },
        CFLLimit: {'dt_max': level_params['dt'], 'dt_min': 1e-6, 'cfl': 0.4},
        StopAtNan: {'thresh': 1e6},
        # SpaceAdaptivity: {'nx_min': 2 * comm.size, 'nz_min': comm.size, 'factor': 2, 'nx_max': 2**10,},
    }

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 2
    sweeper_params['QI'] = 'LU'
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'comm': comm,
        'useGPU': useGPU,
        'Rayleigh': 2e6 / 16,
        'nx': max([2 * comm.size, 2**9]) + 1,
        'nz': max([comm.size, 2**7]),
        'dealiasing': 3 / 2,
    }

    step_params = {}
    step_params['maxiter'] = 3

    controller_params = {}
    controller_params['logger_level'] = 15 if comm.rank == 0 else 40
    controller_params['hook_class'] = [LogToFile, LogGrid, LogAnalysisVariables, LogStepSize]
    controller_params['mssdc_jac'] = False

    description = {}
    description['problem_class'] = RayleighBenardUltraspherical
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

    relaxation_steps = 5
    u0_noise = P.u_exact(t0, seed=comm.rank, noise_level=1e-3)
    uinit = u0_noise
    for _ in range(relaxation_steps):
        uinit = P.solve_system(uinit, dt=0.1)

    # f = P.eval_f(uinit)
    # f_hat = P.transform(f.impl)
    # ip = P.index('p')
    # import matplotlib.pyplot as plt
    # im = plt.imshow(np.log10(np.abs(f_hat[ip])))
    # plt.colorbar(im)
    # plt.show()
    # breakpoint()
    # return None

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    combined_stats = filter_stats(stats, comm=comm)

    if comm.rank == 0:
        path = 'data/RBC-stats'
        with open(f'{path}.pickle', 'wb') as file:
            pickle.dump(combined_stats, file)
    return stats


def plot_RBC(size, quantitiy='T', quantitiy2='vorticity', render=True, start_idx=0):
    import matplotlib.pyplot as plt
    from pySDC.implementations.hooks.log_solution import LogToFileAfterXs as LogToFile
    from pySDC.projects.GPU.hooks.LogGrid import LogGrid
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import gc

    from pySDC.implementations.problem_classes.RayleighBenard import RayleighBenardUltraspherical

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    P = RayleighBenardUltraspherical()

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


def analyse():
    import matplotlib.pyplot as plt

    path = 'data/RBC-stats'
    with open(f'{path}.pickle', 'rb') as file:
        stats = pickle.load(file)

    # plot step size distribution
    dt = get_sorted(stats, type='dt', sortby='time')
    CFL = get_sorted(stats, type='CFL_limit', sortby='time')
    fig, ax = plt.subplots()
    ax.plot([me[0] for me in dt], [me[1] for me in dt], label=r'$\Delta t$')
    ax.plot([me[0] for me in CFL], [me[1] for me in CFL], label='CFL limit')
    ax.legend(frameon=False)
    ax.set_yscale('log')

    # plot Nusselt numbers
    Nusselt = get_sorted(stats, type='Nusselt')
    fig, ax = plt.subplots()
    colors = {
        'V': 'blue',
        't': 'green',
        'b': 'magenta',
        't_no_v': 'green',
        'b_no_v': 'magenta',
    }
    for key in ['V', 't', 'b']:
        ax.plot([me[0] for me in Nusselt], [me[1][f'{key}'] for me in Nusselt], label=f'{key}', color=colors[key])
        ax.axhline(np.mean([me[1][f'{key}'] for me in Nusselt]), color=colors[key], ls='--')
    ax.legend(frameon=False)
    fig.savefig('plots/Nusselt.pdf')

    fig, ax = plt.subplots()
    buoyancy = get_sorted(stats, type='buoyancy_production')
    dissipation = get_sorted(stats, type='viscous_dissipation')
    ax.plot([me[0] for me in buoyancy], [me[1] for me in buoyancy], label=r'buoyancy')
    ax.plot([me[0] for me in dissipation], [me[1] for me in dissipation], label=r'dissipation')
    ax.legend(frameon=False)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=bool)
    parser.add_argument('--plot', type=bool)
    parser.add_argument('--analyse', type=bool)
    parser.add_argument('--useGPU', type=bool)
    parser.add_argument('--render', type=bool)
    parser.add_argument('--startIdx', type=int, default=0)
    parser.add_argument('--np', type=int, default=1)

    args = parser.parse_args()

    if args.run:
        run_RBC(useGPU=args.useGPU)
    if args.plot:
        plot_RBC(size=args.np, render=args.render, start_idx=args.startIdx)
    if args.analyse:
        analyse()