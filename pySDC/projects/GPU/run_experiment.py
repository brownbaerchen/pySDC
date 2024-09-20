def parse_args():
    import argparse

    def cast_to_bool(me):
        return False if me in ['False', '0', 0] else True

    def str_to_procs(me):
        procs = me.split('/')
        assert len(procs) == 3
        return [int(p) for p in procs]

    parser = argparse.ArgumentParser()
    parser.add_argument('--useGPU', type=cast_to_bool, help='Toggle for GPUs', default=False)
    parser.add_argument(
        '--mode', type=str, help='Mode for this script', default='run', choices=['run', 'plot', 'render', 'video']
    )
    parser.add_argument('--config', type=str, help='Configuration to load', default='RBC')
    parser.add_argument('--restart_idx', type=int, help='Restart from file by index', default=0)
    parser.add_argument('--procs', type=str_to_procs, help='Processes in steps/sweeper/space', default='1/1/1')
    parser.add_argument('--res', type=int, help='Space resolution along first axis', default=-1)
    parser.add_argument(
        '--logger_level', type=int, help='Logger level on the first rank in space and in the sweeper', default=15
    )

    return vars(parser.parse_args())


def run_experiment(args, config, **kwargs):
    import pickle

    # from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.stats_helper import filter_stats

    description = config.get_description(
        useGPU=args['useGPU'], MPIsweeper=args['procs'][1] > 1, res=args['res'], **kwargs
    )
    controller_params = config.get_controller_params(logger_level=args['logger_level'])

    # controller = controller_MPI(controller_params, description, config.comms[0])
    assert (
        config.comms[0].size == 1
    ), 'Have not figured out how to do MPI controller with GPUs yet because I need NCCL for that!'
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # u0, t0 = config.get_initial_condition(controller.S.levels[0].prob, restart_idx=args['restart_idx'])
    u0, t0 = config.get_initial_condition(controller.MS[0].levels[0].prob, restart_idx=args['restart_idx'])

    uend, stats = controller.run(u0=u0, t0=t0, Tend=config.Tend)

    combined_stats = filter_stats(stats, comm=config.comm_world)

    if config.comm_world.rank == config.comm_world.size - 1:
        path = f'data/{config.get_path()}-stats.pickle'
        with open(path, 'wb') as file:
            pickle.dump(combined_stats, file)

    return uend


def plot_experiment(args, config):  # pragma: no cover
    import gc
    import matplotlib.pyplot as plt

    description = config.get_description()

    P = description['problem_class'](**description['problem_params'])

    comm = config.comm_world

    for idx in range(args['restart_idx'], 9999, comm.size):
        try:
            fig = config.plot(P, idx + comm.rank, args['procs'])
        except FileNotFoundError:
            break

        path = f'simulation_plots/{config.get_path(ranks=[0,0,0])}-{idx+comm.rank:06d}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'{comm.rank} Stored figure {path!r}', flush=True)

        if args['mode'] == 'render':
            plt.pause(1e-9)

        plt.close(fig)
        del fig
        gc.collect()


def make_video(args, config):  # pragma: no cover
    comm = config.comm_world
    if comm.rank > 0:
        return None

    import subprocess

    path = f'simulation_plots/{config.get_path(ranks=[0,0,0])}-%06d.png'
    path_target = f'videos/{args["config"]}.mp4'

    cmd = f'ffmpeg -i {path} -pix_fmt yuv420p -r 9 -s 2048:1536 {path_target}'.split()

    subprocess.run(cmd)


if __name__ == '__main__':
    from pySDC.projects.GPU.configs.base_config import get_config

    args = parse_args()

    config = get_config(args)

    if args['mode'] == 'run':
        run_experiment(args, config)
    elif args['mode'] in ['plot', 'render']:  # pragma: no cover
        plot_experiment(args, config)
    elif args['mode'] == 'video':  # pragma: no cover
        make_video(args, config)
