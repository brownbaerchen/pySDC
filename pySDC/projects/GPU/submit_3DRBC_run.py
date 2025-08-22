import numpy as np
import argparse
from pySDC.projects.GPU.etc.generate_jobscript import write_jobscript


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
        '--mode',
        type=str,
        help='Mode for this script',
        default=None,
        choices=['run', 'plot', 'render', 'plot_series', 'video'],
    )
    parser.add_argument('--config', type=str, help='Configuration to load', default=None)
    parser.add_argument('--restart_idx', type=int, help='Restart from file by index', default=0)
    parser.add_argument('--procs', type=str_to_procs, help='Processes in steps/sweeper/space', default='1/1/1')
    parser.add_argument('--res', type=int, help='Space resolution along first axis', default=-1)
    parser.add_argument('--dt', type=float, help='(Starting) Step size', default=-1)
    parser.add_argument(
        '--logger_level', type=int, help='Logger level on the first rank in space and in the sweeper', default=15
    )
    parser.add_argument('--tasks_per_node', type=int)
    parser.add_argument('--time', type=int)
    parser.add_argument('--partition', type=str, default='batch')
    parser.add_argument('--cluster', type=str, default='JUSUF')
    parser.add_argument('-o', type=str, help='output path', default='./')

    return vars(parser.parse_args())


def run(args, **kwargs):
    res = args['res']
    config = args['config']
    dt = args['dt']
    procs = args['procs']
    tasks_per_node = args['tasks_per_node']
    OMP_NUM_THREADS = 1

    sbatch_options = [
        f'-n {np.prod(procs)}',
        f'-p {args["partition"]}',
        f'--tasks-per-node={tasks_per_node}',
        f'--time={args["time"]}:00:00',
    ]
    srun_options = [f'--tasks-per-node={tasks_per_node}']
    if args['useGPU']:
        srun_options += [f'--cpus-per-task={OMP_NUM_THREADS}', '--gpus-per-task=1']
        sbatch_options += [f'--cpus-per-task={OMP_NUM_THREADS}', '--gpus-per-task=1']

    procs = (''.join(f'{me}/' for me in procs))[:-1]
    command = f'run_experiment.py --mode=run --res={res} --dt={dt} --config={args["config"]} --procs={procs}'

    if args['restart_idx'] != 0:
        command += f' --restart_idx={args["restart_idx"]}'

    if args["useGPU"]:
        command += ' --useGPU=True'

    write_jobscript(
        sbatch_options,
        srun_options,
        command,
        args['cluster'],
        name=f'{args["config"]}_{res}_{dt}',
        OMP_NUM_THREADS=OMP_NUM_THREADS,
        **kwargs,
    )


if __name__ == '__main__':
    args = parse_args()
    run(args)
