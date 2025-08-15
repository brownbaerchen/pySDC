from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.projects.GPU.run_experiment import parse_args
from pySDC.helpers.fieldsIO import FieldsIO
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpi4py import MPI
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
import os

PADDING = 1
LIVE = False
IDX = -12


args = parse_args()

comm = MPI.COMM_WORLD

config = get_config(args)


fname = config.get_file_name()
start = fname.index('data')
_name = fname[start + 5 : -6]
base_path = f'simulation_plots/{_name}'
print(f'Writing plots to directory {base_path!r}')
os.makedirs(base_path, exist_ok=True)

desc = config.get_description(**args)
P = desc['problem_class'](**{**desc['problem_params'], 'spectral_space': False, 'comm': comm})
P.setUpFieldsIO()
xp = P.xp
X, Y = P.X[:, :, IDX], P.Y[:, :, IDX]


fname = config.get_file_name()

start = fname.index('data')
path = f'{base_path}/{fname[start + 5:-6]}.pickle'

data = FieldsIO.fromFile(fname)

plt.rcParams['figure.constrained_layout.use'] = True
fig, ax = plt.subplots(figsize=(6, 5))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.03)


r = range(args['restart_idx'], data.nFields)
if P.comm.rank == 0:
    r = tqdm(r)
for i in r:
    _t, u = data.readField(i)

    if PADDING != 1:
        u_hat = P.transform(u)
        u_pad = P.itransform(u_hat, padding=(PADDING, PADDING, 1)).real
        x = P.xp.linspace(0, P.axes[0].L, u_pad.shape[1])
        y = P.xp.linspace(0, P.axes[1].L, u_pad.shape[2])
        X, Y = xp.meshgrid(x, y)
        im = ax.pcolormesh(X, Y, u_pad[P.index('T'), :, :, IDX], cmap='inferno', rasterized=True)
    else:
        im = ax.pcolormesh(X, Y, u[P.index('T'), :, :, IDX], cmap='inferno', rasterized=True)
    ax.set_title(f't={_t:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect(1.0)
    fig.colorbar(im, cax)
    fig.savefig(f'{base_path}/{_name}_{i:06d}.png', dpi=300)

    if LIVE:
        plt.pause(1e-9)
    ax.cla()
