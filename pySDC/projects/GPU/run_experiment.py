def parse_args():
    import argparse

    cast_to_bool = lambda me: False if me == 'False' else True

    def str_to_procs(me):
        procs = me.split('/')
        assert len(procs) == 3
        return [int(p) for p in procs]

    parser = argparse.ArgumentParser()
    parser.add_argument('--useGPU', type=cast_to_bool, help='Toggle for GPUs', default=True)
    parser.add_argument(
        '--mode', type=str, help='Mode for this script', default='run', choices=['run', 'plot', 'render']
    )
    parser.add_argument('--config', type=str, help='Configuration to load', default='RBC')
    parser.add_argument('--restart_idx', type=int, help='Restart from file by index', default=0)
    parser.add_argument('--procs', type=str_to_procs, help='Processes in steps/sweeper/space', default='1/1/1')
    parser.add_argument(
        '--logger_level', type=int, help='Logger level on the first rank in space and in the sweeper', default=15
    )

    return parser.parse_args()


def run_experiment():
    import pickle
    from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
    from pySDC.helpers.stats_helper import filter_stats
    from pySDC.projects.GPU.configs import get_config

    args = parse_args()

    config = get_config(args.config, n_procs_list=args.procs)

    description = config.get_description()
    controller_params = config.get_controller_params()

    controller = controller_MPI(controller_params, description, config.comms[0])

    u0, t0 = config.get_initial_condition(controller.S.levels[0].prob, restart_idx=args.restart_idx)

    uend, stats = controller.run(u0=u0, t0=t0, Tend=config.Tend)

    combined_stats = filter_stats(stats, comm=config.comm_world)

    if config.comm_world.rank == config.comm_world.size - 1:
        path = f'data/{config.get_path()}-stats.pickle'
        with open(path, 'wb') as file:
            pickle.dump(combined_stats, file)


def plot_experiment():
    from pySDC.projects.GPU.configs import get_config
    import gc

    args = parse_args()
    config = get_config(args.config, n_procs_list=args.procs)

    description = config.get_description()

    P = description['problem_class'](**description['problem_params'])

    comm = config.comm_world

    for idx in range(args.restart_idx, 9999, comm.size):
        fig = config.plot(P, idx + comm.rank, args.procs)
        import matplotlib.pyplot as plt

        path = f'config.get_path(ranks=[0,0,0])-{idx:06d}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'{comm.rank} Stored figure {path!r}', flush=True)

        if args.mode == 'render':
            plt.pause(1e-9)

        plt.close(fig)
        del fig
        gc.collect()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'run':
        run_experiment()
    elif args.mode in ['plot', 'render']:
        plot_experiment()
