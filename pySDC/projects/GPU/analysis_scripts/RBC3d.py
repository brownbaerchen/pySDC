from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.projects.GPU.run_experiment import parse_args
from pySDC.helpers.fieldsIO import FieldsIO
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpi4py import MPI
import numpy as np
import pickle
import os

PLOT = False
BASE_PATH = 'data/RBC_time_averaged'
PADDING = 1


os.makedirs(BASE_PATH, exist_ok=True)

args = parse_args()

comm = MPI.COMM_WORLD

config = get_config(args)

desc = config.get_description(**args)
P = desc['problem_class'](**{**desc['problem_params'], 'spectral_space': False, 'comm': comm})
P.setUpFieldsIO()
xp = P.xp

fname = config.get_file_name()

start = fname.index('data')
path = f'{BASE_PATH}/{fname[start + 5:-6]}.pickle'

data = FieldsIO.fromFile(fname)

Nu = {'V': [], 'b': [], 't': []}
t = []
T = []
spectrum = []
Re = []
CFL = []

X, Y = P.X[:, :, -1], P.Y[:, :, -1]

r = range(args['restart_idx'], data.nFields)
if P.comm.rank == 0:
    r = tqdm(r)
for i in r:
    _t, u = data.readField(i)

    _Nu = P.compute_Nusselt_numbers(u)
    if any(me > 1e3 for me in _Nu.values()):
        continue

    for key in Nu.keys():
        Nu[key].append(_Nu[key])

    t.append(_t)

    if PLOT:
        if PADDING != 1:
            u_hat = P.transform(u)
            u_pad = P.itransform(u_hat, padding=(PADDING, PADDING, 1)).real
            x = P.xp.linspace(0, P.axes[0].L, u_pad.shape[1])
            y = P.xp.linspace(0, P.axes[1].L, u_pad.shape[2])
            X, Y = xp.meshgrid(x, y)
            plt.pcolormesh(X, Y, u_pad[P.index('T'), :, :, -1], cmap='inferno', rasterized=True)
        else:
            plt.pcolormesh(X, Y, u[P.index('T'), :, :, -1], cmap='inferno', rasterized=True)
        plt.title(f't={_t:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(1e-9)

    T.append(P.get_vertical_temperature_profile(u))
    Re.append(P.get_Reynolds_number(u))
    CFL.append(P.get_CFL_limit(u))

    k, s = P.get_frequency_spectrum(u)
    spectrum.append(s[0, 0])
    # plt.loglog(k[s[0,0]>1e-15], s[0,0][s[0,0]>1e-15])
    # # plt.ylim(1e-10, 1e1)
    # plt.pause(1e-9)
    # plt.clf()


t = xp.array(t)


# plt.plot(t, Re)
# plt.show()
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for key in Nu.keys():
    axs[0].plot(t, Nu[key], label=f'$Nu_{{{key}}}$')
    if config.converged > 0:
        axs[0].axvline(config.converged, color='black')
axs[0].set_ylabel('$Nu$')
axs[0].set_xlabel('$t$')
axs[0].legend(frameon=False)

# compute differences in Nusselt numbers
avg_Nu = {}
std_Nu = {}
for key in Nu.keys():
    _Nu = [Nu[key][i] for i in range(len(Nu[key])) if t[i] > config.converged]
    avg_Nu[key] = xp.mean(_Nu)
    std_Nu[key] = xp.std(_Nu)
avg_Re = xp.mean([Re[i] for i in range(len(Re)) if t[i] > config.converged])

rel_error = {key: abs(avg_Nu[key] - avg_Nu['V']) / avg_Nu['V'] for key in ['t', 'b']}
if comm.rank == 0:
    print(
        f'With Ra={P.Rayleigh:.0e} got Nu={avg_Nu["V"]:.2f}+-{std_Nu["V"]:.2f} with error at top of {rel_error["t"]:.2e} and {rel_error["b"]:.2e} at the bottom and Re={avg_Re:.2e}'
    )


# average T
_T = [T[i] for i in range(len(T)) if t[i] >= config.converged]
avg_T = P.xp.mean(_T, axis=0)
axs[1].axvline(0.5, color='black')
axs[1].plot(avg_T, P.axes[-1].get_1dgrid())
axs[1].set_xlabel('$T$')
axs[1].set_ylabel('$z$')

# spectrum
_s = xp.array(spectrum)
avg_spectrum = xp.mean(_s[t >= config.converged], axis=0)
axs[2].loglog(k[avg_spectrum > 1e-15], avg_spectrum[avg_spectrum > 1e-15])
axs[2].set_xlabel('$k$')
axs[2].set_ylabel(r'$\|\hat{u}_x\|$')

if P.comm.rank == 0:
    write_data = {
        't': t,
        'Nu': Nu,
        'avg_Nu': avg_Nu,
        'std_Nu': std_Nu,
        'T_vertical': avg_T,
        'z': P.axes[-1].get_1dgrid(),
        'k': k,
        'spectrum': avg_spectrum,
    }
    with open(path, 'wb') as file:
        pickle.dump(write_data, file)
        print(f'Wrote data to file {path!r}')

    fig.tight_layout()
    fig.save_fig(f'{BASE_PATH}/{fname[start + 5:-6]}.pdf')
    plt.show()
