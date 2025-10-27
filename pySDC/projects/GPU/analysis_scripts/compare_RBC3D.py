import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def get_pySDC_data(Ra, RK=False, res=-1, dt=-1, config_name='RBC3DG4'):
    assert type(Ra) == str

    base_path = 'data/RBC_time_averaged'

    if RK:
        config_name = f'{config_name}RK'

    path = f'{base_path}/{config_name}Ra{Ra}-res{res}-dt{dt:.0e}.pickle'
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def get_reference_Nu_Ra_scaling_G05():
    data = np.loadtxt("data/rbc_comparison_data/0_auxDat/Nu_Ra_Niemela.dat")
    Ra = data[:, 0]
    Nu = data[:, 1]
    return Ra, Nu


def get_Nek5000_Data(Ra):
    assert type(Ra) == str
    base_path = 'data/rbc_comparison_data/'

    if Ra == '1e5':
        dir_name = '1_1e5'
        start_time = 3500
        nelZ = 64
        nPoly = 5
    elif Ra == '1e6':
        dir_name = '2_1e6'
        start_time = 3500
        nelZ = 64
        nPoly = 7
    elif Ra == '1e7':
        dir_name = '3_1e7'
        start_time = 3100
        nelZ = 64
        nPoly = 9
    elif Ra == '1e8':
        dir_name = '4_1e8'
        start_time = 3000
        nelZ = 96
        nPoly = 7
    elif Ra == '1e9':
        dir_name = '5_1e9'
        start_time = 4000
        nelZ = 96
        nPoly = 9
    elif Ra == '1e10':
        dir_name = '6_1e10'
        start_time = 1700
        nelZ = 200
        nPoly = 7
    elif Ra == '1e11':
        dir_name = '7_1e11'
        start_time = 260
        nelZ = 256
        nPoly = 7
    else:
        raise

    path = f'{base_path}/{dir_name}'
    data = {}

    # get averaged data
    avg = np.load(f'{path}/average.npy')
    avg_Nu = np.mean(avg[avg[:, 0] > start_time, 3])
    data['Nu'] = avg_Nu
    data['std_Nu'] = np.std(avg[avg[:, 0] > start_time, 3])

    # get profile data
    profiles = np.load(f'{path}/profile.npy')
    nzPts = nelZ * nPoly + 1
    nSnap = int(profiles.shape[0] / nzPts)
    tVal = profiles[:, 0].reshape((nSnap, nzPts))[:, 0]
    tInterval = tVal[-1] - tVal[0]

    data['z'] = profiles[:nzPts, 1]
    data['profile_T'] = profiles[:, 3].reshape((nSnap, nzPts))

    tRMS = profiles[:, 4].reshape((nSnap, nzPts))
    data['rms_profile_T'] = np.sqrt(integrate.simpson(tRMS**2, tVal, axis=0) / tInterval)

    return data


def plot_Ra_Nusselt_scaling():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3))
    NuRa_ax = axs[0]
    prof_ax = axs[1]
    rms_ax = axs[2]
    spectrum_ax = axs[3]

    for Ra, Ra_str in zip([1e5, 1e6, 1e7, 1e8], ['1e5', '1e6', '1e7', '1e8']):
        if Ra > 1e6:
            data_pySDC = get_pySDC_data(Ra_str, config_name='RBC3DG4R4')
        else:
            data_pySDC = get_pySDC_data(Ra_str)
        data_Nek5000 = get_Nek5000_Data(Ra_str)

        # NuRa_ax.scatter(Ra, data_Nek5000['Nu'], color='black')
        NuRa_ax.errorbar(Ra, data_Nek5000['Nu'], yerr=data_Nek5000['std_Nu'], color='black', fmt='o')
        NuRa_ax.errorbar(Ra, data_pySDC['avg_Nu']['V'], yerr=data_pySDC['std_Nu']['V'], color='red', fmt='.')

        prof_ax.plot(data_Nek5000['profile_T'].mean(axis=0), data_Nek5000['z'])  # , label=f'Nek5000 Ra={Ra:.0e}')
        prof_ax.scatter(data_pySDC['profile_T'], data_pySDC['z'])  # , label=f'pySDC Ra={Ra:.0e}')

        rms_ax.plot(data_Nek5000['rms_profile_T'], data_Nek5000['z'], label=f'Nek5000 Ra={Ra:.0e}')
        # rms_ax.axhline(data_Nek5000['z'][np.argmax(data_Nek5000['rms_profile_T'][:int(len(data_Nek5000['z'])/2)])])

        rms_ax.scatter(data_pySDC['rms_profile_T'], data_pySDC['z'], label=f'pySDC Ra={Ra:.0e}')
        # shift = int(len(data_pySDC['z']) / 2)
        # rms_ax.axhline(data_pySDC['z'][shift + np.argmax(data_pySDC['rms_profile_T'][shift:])], ls='--', color='red')

        k = data_pySDC['k'] + 1
        spectrum = np.array(data_pySDC['spectrum'])
        u_spectrum = np.mean(spectrum, axis=0)[1]
        idx = data_pySDC['res_in_boundary_layer']
        _s = u_spectrum[idx]
        spectrum_ax.loglog(
            k[_s > 1e-16], _s[_s > 1e-16]  # , color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
        )

    NuRa_ax.errorbar([], [], color='red', fmt='.', label='pySDC')
    NuRa_ax.scatter([], [], color='black', label='Nek5000')
    NuRa_ax.legend(frameon=False)

    prof_ax.plot([], [], color='black', label='Nek5000')
    prof_ax.scatter([], [], color='black', label='pySDC')
    prof_ax.legend(frameon=False)
    # prof_ax.legend(frameon=False)
    prof_ax.set_xlabel('T')
    prof_ax.set_ylabel('z')
    prof_ax.set_xlim((0.47, 1.03))
    prof_ax.set_ylim((-0.01, 0.30))

    rms_ax.set_xlabel(r'$T_\text{rms}$')
    rms_ax.set_ylabel('z')
    rms_ax.set_xlim((-0.01, 0.177))
    rms_ax.set_ylim((-0.01, 0.23))

    NuRa_ax.set_xlabel('Ra')
    NuRa_ax.set_ylabel('Nu')
    NuRa_ax.set_yscale('log')
    NuRa_ax.set_xscale('log')

    spectrum_ax.set_xlabel('$k$')
    spectrum_ax.set_ylabel(r'$\|\hat{u}_x\|$')

    for ax in axs:
        ax.set_box_aspect(1)

    fig.tight_layout()
    fig.savefig('./data/RBC_time_averaged/Nek5000_pySDC_comparison.pdf')


def compare_Nusselt_over_time1e7_old():
    fig, ax = plt.subplots()
    Ra = '1e7'

    data = []
    labels = []
    linestyles = []

    data.append(get_pySDC_data(Ra, res=96, dt=0.01))
    labels.append('SDC, res=96, dt=0.01')
    linestyles.append('-')

    data.append(get_pySDC_data(Ra, res=96, dt=0.009))
    labels.append('SDC, res=96, dt=0.009')
    linestyles.append('-')

    # data.append(get_pySDC_data(Ra, res=48, dt=0.02))
    # labels.append(['SDC, res=48, dt=0.02'])

    data.append(get_pySDC_data(Ra, RK=True, res=96, dt=0.008))
    labels.append('RK, res=96, dt=0.008')
    linestyles.append('--')

    data.append(get_pySDC_data(Ra, RK=True, res=96, dt=0.006))
    labels.append('RK, res=96, dt=0.006')
    linestyles.append('--')

    data.append(get_pySDC_data(Ra, RK=True, res=96, dt=0.003))
    labels.append('RK, res=96, dt=0.003')
    linestyles.append('--')

    for dat, label, linestyle in zip(data, labels, linestyles):
        Nu = dat['Nu']['V']
        ax.plot(dat['t'], dat['Nu']['V'], label=label, ls=linestyle)
        print(label, np.mean(Nu), np.std(Nu))

    ax.legend(frameon=False)
    ax.set_xlabel('t')
    ax.set_ylabel('Nu')


def compare_Nusselt_over_time1e5_old():
    fig, axs = plt.subplots(2, 1, sharex=True)
    Ra = '1e5'
    res = 32

    _, prof_ax = plt.subplots()

    data = []
    labels = []
    linestyles = []

    data.append(get_pySDC_data(Ra, res=-1, dt=-1))
    labels.append('SDC, dt=0.08')
    linestyles.append('-')

    ref_data = get_pySDC_data(Ra, res=res, dt=0.002)

    for dt in [0.06, 0.04, 0.02, 0.01, 0.005]:
        # for dt in [0.04, 0.02, 0.01, 0.005, 0.002]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt))
        labels.append(f'SDC, dt={dt:.3f}')
        linestyles.append('-')

    # ----------------- RK ------------------------

    for dt in [0.07, 0.06, 0.04, 0.02, 0.01, 0.005]:
        # for dt in [0.04, 0.02, 0.01, 0.005, 0.002]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt, RK=True))
        labels.append(f'RK, dt={dt:.3f}')
        linestyles.append('--')

    for dat, label, linestyle in zip(data, labels, linestyles):
        Nu = np.array(dat['Nu']['V'])
        t = dat['t']
        Nu_ref = np.array(ref_data['Nu']['V'])
        t_i, Nu_i = interpolate_NuV_to_reference_times(dat, ref_data)
        axs[0].plot(t, Nu, label=label, ls=linestyle)

        prof_ax.plot(dat['rms_profile_T'], dat['z'])

        error = np.maximum.accumulate(np.abs(Nu_ref[: Nu_i.shape[0]] - Nu_i) / np.abs(Nu_ref[: Nu_i.shape[0]]))
        axs[1].plot(t_i, error, label=label, ls=linestyle)

        # compute mean Nu
        mask = np.logical_and(t >= 50, t <= 200)
        Nu_mean = np.mean(Nu[mask])
        Nu_std = np.std(Nu[mask])

        if any(error > 1e-2):
            deviates = min(t_i[error > 1e-2])
            last_line = axs[0].get_lines()[-1]
            for _ax in axs:
                _ax.axvline(deviates, color=last_line.get_color(), ls=last_line.get_linestyle())
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}, deviates more than 1% from t={deviates:.2f}')
        else:
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}')

        axs[1].set_yscale('log')
        axs[1].set_ylim((1e-5, 1))
        # print(dat['t'][:12])

    axs[1].legend(frameon=True)
    axs[1].set_xlabel('t')
    axs[0].set_ylabel('Nu')
    axs[1].set_ylabel('Error in Nu')


def interpolate_NuV_to_reference_times(data, reference_data, order=12):
    from qmat.lagrange import getSparseInterpolationMatrix

    t_in = np.array(data['t'])
    t_out = np.array([me for me in reference_data['t'] if me <= max(t_in)])

    interpolation_matrix = getSparseInterpolationMatrix(t_in, t_out, order=order)
    return interpolation_matrix @ t_in, interpolation_matrix @ data['Nu']['V']


def compare_Nusselt_over_time1e5():
    fig, Nu_ax = plt.subplots()
    _, axs = plt.subplots(1, 3, figsize=(10, 3))
    prof_ax = axs[0]
    rms_ax = axs[1]
    spectrum_ax = axs[2]

    Ra = '1e5'
    res = 32

    data = []
    labels = []
    linestyles = []

    ref_data = get_pySDC_data(Ra, res=res, dt=0.01, config_name='RBC3DG4R4')

    _, ting_ax = plt.subplots()

    for dt in [0.06, 0.02, 0.01]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt, config_name='RBC3DG4R4'))
        labels.append(f'SDC, dt={dt:.4f}')
        linestyles.append('-')

    # ----------------- RK ------------------------

    for dt in [0.05, 0.04, 0.02, 0.01, 0.005]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt, RK=True, config_name='RBC3DG4R4'))
        labels.append(f'RK, dt={dt:.3f}')
        linestyles.append('--')

    # ----------------- Euler ---------------------

    for dt in [0.02, 0.005]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt, config_name='RBC3DG4R4Euler'))
        labels.append(f'Euler, dt={dt:.3f}')
        linestyles.append('-.')

    for dat, label, linestyle in zip(data, labels, linestyles):
        Nu = np.array(dat['Nu']['V'])
        t = dat['t']
        Nu_ref = np.array(ref_data['Nu']['V'])
        t_i, Nu_i = interpolate_NuV_to_reference_times(dat, ref_data)
        Nu_ax.plot(t, Nu, label=label, ls=linestyle)

        error = np.maximum.accumulate(np.abs(Nu_ref[: Nu_i.shape[0]] - Nu_i) / np.abs(Nu_ref[: Nu_i.shape[0]]))

        # compute mean Nu
        mask = np.logical_and(t >= 20, t <= 200)
        Nu_mean = np.mean(Nu[mask])
        Nu_std = np.std(Nu[mask])

        last_line = Nu_ax.get_lines()[-1]
        if any(error > 1e-2):
            deviates = min(t_i[error > 1e-2])
            Nu_ax.axvline(deviates, color=last_line.get_color(), ls=last_line.get_linestyle())
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}, deviates more than 1% from t={deviates:.2f}')
        else:
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}')

        k = dat['k']
        spectrum = np.array(dat['spectrum'])
        u_spectrum = np.mean(spectrum, axis=0)[1]
        idx = dat['res_in_boundary_layer']
        _s = u_spectrum[idx]
        spectrum_ax.loglog(
            k[_s > 1e-16], _s[_s > 1e-16], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
        )

        prof_ax.plot(dat['profile_T'], dat['z'], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label)
        rms_ax.plot(
            dat['rms_profile_T'], dat['z'], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
        )
        if 'dissipation' in dat.keys():

            mean_ke = np.mean([me[1] for me in dat['dissipation']])
            print(
                f'Energy error: {(dat["dissipation"][-1][1] - np.sum([me[0] for me in dat["dissipation"]])/2)/mean_ke:.2e}'
            )
            ting_ax.plot(t, np.cumsum([me[0] for me in dat['dissipation']]) / 2, label=label)
            ting_ax.plot(t, [me[1] for me in dat['dissipation']], label=label, ls='--')
            ting_ax.legend()

    Nu_ax.legend(frameon=True)
    Nu_ax.set_xlabel('t')
    Nu_ax.set_ylabel('Nu')

    spectrum_ax.legend(frameon=False)
    spectrum_ax.set_xlabel('$k$')
    spectrum_ax.set_ylabel(r'$\|\hat{u}_x\|$')


def compare_Nusselt_over_time1e6():
    fig, Nu_ax = plt.subplots()
    _, axs = plt.subplots(1, 3, figsize=(10, 3))
    prof_ax = axs[0]
    rms_ax = axs[1]
    spectrum_ax = axs[2]

    Ra = '1e6'
    res = 48

    data = []
    labels = []
    linestyles = []

    ref_data = get_pySDC_data(Ra, res=res, dt=0.002)

    for dt in [0.04, 0.02, 0.01, 0.005, 0.002]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt))
        labels.append(f'SDC, dt={dt:.4f}')
        linestyles.append('-')

    # # ----------------- RK ------------------------

    for dt in [0.03, 0.02, 0.01, 0.005, 0.002]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt, RK=True))
        labels.append(f'RK, dt={dt:.3f}')
        linestyles.append('--')

    data.append(get_pySDC_data(Ra, res=res, dt=0.02, config_name='RBC3DG4R4'))
    labels.append(f'SDC, R4, dt={0.02:.4f}')
    linestyles.append(':')

    for dat, label, linestyle in zip(data, labels, linestyles):
        Nu = np.array(dat['Nu']['V'])
        t = dat['t']
        Nu_ref = np.array(ref_data['Nu']['V'])
        t_i, Nu_i = interpolate_NuV_to_reference_times(dat, ref_data)
        Nu_ax.plot(t, Nu, label=label, ls=linestyle)

        error = np.maximum.accumulate(np.abs(Nu_ref[: Nu_i.shape[0]] - Nu_i) / np.abs(Nu_ref[: Nu_i.shape[0]]))

        # compute mean Nu
        mask = np.logical_and(t >= 20, t <= 200)
        Nu_mean = np.mean(Nu[mask])
        Nu_std = np.std(Nu[mask])

        last_line = Nu_ax.get_lines()[-1]
        if any(error > 1e-2):
            deviates = min(t_i[error > 1e-2])
            Nu_ax.axvline(deviates, color=last_line.get_color(), ls=last_line.get_linestyle())
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}, deviates more than 1% from t={deviates:.2f}')
        else:
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}')

        k = dat['k']
        spectrum = np.array(dat['spectrum'])
        u_spectrum = np.mean(spectrum, axis=0)[1]
        idx = dat['res_in_boundary_layer']
        _s = u_spectrum[idx]
        spectrum_ax.loglog(
            k[_s > 1e-16], _s[_s > 1e-16], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
        )

        prof_ax.plot(dat['profile_T'], dat['z'], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label)
        rms_ax.plot(
            dat['rms_profile_T'], dat['z'], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
        )

    Nu_ax.legend(frameon=True)
    Nu_ax.set_xlabel('t')
    Nu_ax.set_ylabel('Nu')

    spectrum_ax.legend(frameon=False)
    spectrum_ax.set_xlabel('$k$')
    spectrum_ax.set_ylabel(r'$\|\hat{u}_x\|$')


def compare_Nusselt_over_time1e7():
    fig, Nu_ax = plt.subplots()
    _, axs = plt.subplots(1, 3, figsize=(10, 3))
    prof_ax = axs[0]
    rms_ax = axs[1]
    spectrum_ax = axs[2]

    Ra = '1e7'
    res = 96

    data = []
    labels = []
    linestyles = []

    ref_data = get_pySDC_data(Ra, res=res, dt=0.009)

    for dt in [0.01, 0.009]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt))
        labels.append(f'SDC, dt={dt:.4f}')
        linestyles.append('-')

    # # ----------------- RK ------------------------

    for dt in [0.008, 0.006, 0.003]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt, RK=True))
        labels.append(f'RK, dt={dt:.3f}')
        linestyles.append('--')

    for dat, label, linestyle in zip(data, labels, linestyles):
        Nu = np.array(dat['Nu']['V'])
        t = dat['t']
        Nu_ref = np.array(ref_data['Nu']['V'])
        t_i, Nu_i = interpolate_NuV_to_reference_times(dat, ref_data)
        Nu_ax.plot(t, Nu, label=label, ls=linestyle)

        error = np.maximum.accumulate(np.abs(Nu_ref[: Nu_i.shape[0]] - Nu_i) / np.abs(Nu_ref[: Nu_i.shape[0]]))

        # compute mean Nu
        mask = np.logical_and(t >= 20, t <= 200)
        Nu_mean = np.mean(Nu[mask])
        Nu_std = np.std(Nu[mask])

        last_line = Nu_ax.get_lines()[-1]
        if any(error > 1e-2):
            deviates = min(t_i[error > 1e-2])
            Nu_ax.axvline(deviates, color=last_line.get_color(), ls=last_line.get_linestyle())
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}, deviates more than 1% from t={deviates:.2f}')
        else:
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}')

        k = dat['k']
        spectrum = np.array(dat['spectrum'])
        try:
            u_spectrum = np.mean(spectrum, axis=0)[1]
            # u_spectrum = spectrum[0][1]
            idx = dat['res_in_boundary_layer']
            _s = u_spectrum[idx]
            spectrum_ax.loglog(
                k[_s > 1e-16], _s[_s > 1e-16], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
            )
        except:
            pass

        prof_ax.plot(dat['profile_T'], dat['z'], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label)
        rms_ax.plot(
            dat['rms_profile_T'], dat['z'], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
        )

    Nu_ax.legend(frameon=True)
    Nu_ax.set_xlabel('t')
    Nu_ax.set_ylabel('Nu')

    spectrum_ax.legend(frameon=False)
    spectrum_ax.set_xlabel('$k$')
    spectrum_ax.set_ylabel(r'$\|\hat{u}_x\|$')


def compare_Nusselt_over_time1e8():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3))
    Nu_ax = axs[0]
    prof_ax = axs[1]
    rms_ax = axs[2]
    spectrum_ax = axs[3]

    Ra = '1e8'
    res = 96

    data = []
    labels = []
    linestyles = []

    ref_data = get_pySDC_data(Ra, res=res, dt=6e-3)

    for dt in [6e-3]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt))
        labels.append(f'SDC, dt={dt:.4f}')
        linestyles.append('-')

    # # ----------------- RK ------------------------

    for dt in [0.005]:
        data.append(get_pySDC_data(Ra, res=res, dt=dt, RK=True))
        labels.append(f'RK, dt={dt:.3f}')
        linestyles.append('--')

    for dat, label, linestyle in zip(data, labels, linestyles):
        Nu = np.array(dat['Nu']['V'])
        t = dat['t']
        Nu_ref = np.array(ref_data['Nu']['V'])
        t_i, Nu_i = interpolate_NuV_to_reference_times(dat, ref_data)
        Nu_ax.plot(t, Nu, label=label, ls=linestyle)

        error = np.maximum.accumulate(np.abs(Nu_ref[: Nu_i.shape[0]] - Nu_i) / np.abs(Nu_ref[: Nu_i.shape[0]]))

        # compute mean Nu
        mask = np.logical_and(t >= 20, t <= 200)
        Nu_mean = np.mean(Nu[mask])
        Nu_std = np.std(Nu[mask])

        last_line = Nu_ax.get_lines()[-1]
        if any(error > 1e-2):
            deviates = min(t_i[error > 1e-2])
            Nu_ax.axvline(deviates, color=last_line.get_color(), ls=last_line.get_linestyle())
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}, deviates more than 1% from t={deviates:.2f}')
        else:
            print(f'{label} Nu={Nu_mean:.3f}+={Nu_std:.3f}')

        k = dat['k']
        spectrum = np.array(dat['spectrum'])
        u_spectrum = np.mean(spectrum, axis=0)[1]
        # u_spectrum = spectrum[0, 0, :]
        idx = dat['res_in_boundary_layer']
        _s = u_spectrum[idx]
        spectrum_ax.loglog(
            k[_s > 1e-16], _s[_s > 1e-16], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
        )

        prof_ax.plot(dat['profile_T'], dat['z'], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label)
        rms_ax.plot(
            dat['rms_profile_T'], dat['z'], color=last_line.get_color(), ls=last_line.get_linestyle(), label=label
        )

    Nu_ax.legend(frameon=True)
    Nu_ax.set_xlabel('t')
    Nu_ax.set_ylabel('Nu')

    spectrum_ax.legend(frameon=False)
    spectrum_ax.set_xlabel('$k$')
    spectrum_ax.set_ylabel(r'$\|\hat{u}_x\|$')


def plot_thibaut_stuff():
    fig, ax = plt.subplots()

    data = get_pySDC_data('1e7', res=96, dt=0.009)

    Nu = data['Nu']['V']
    t = data['t']
    avg_Nu = np.array([np.mean(Nu[40 : 40 + i + 1]) for i in range(len(Nu[40:]))])
    Delta_Nu = np.array([abs(avg_Nu[i + 1] - avg_Nu[i]) for i in range(len(avg_Nu) - 1)])
    # ax.plot(data['t'][40:-1], Delta_Nu / avg_Nu[:-1])
    # ax.plot(data['t'], np.abs(avg_Nu - avg_Nu[-1]) / avg_Nu[-1])
    ax.plot(t[40:], avg_Nu)
    ax.plot(t[40:], Nu[40:])
    ax.set_xlabel('t')
    # ax.set_ylabel('Delta Nu / Nu')


def plot_spectrum_over_time1e6R4():
    fig, ax = plt.subplots()

    # data = get_pySDC_data('1e6', res=48, dt=0.02, config_name='RBC3DG4')
    # data = get_pySDC_data('1e7', res=96, dt=0.009, config_name='RBC3DG4')
    data = get_pySDC_data('1e7', res=64, dt=0.005, config_name='RBC3DG4R4D42')
    # data = get_pySDC_data('1e8', res=96, dt=0.005, config_name='RBC3DG4R4')
    # data = get_pySDC_data('1e8', res=256, dt=0.002, config_name='RBC3DG4R4RK')

    s = data['spectrum']
    t = data['t']
    k = data['k']

    for i in range(len(s)):
        # for i in [0, 3, 10, 20, 40, 80, -1]:
        # for i in [0, 5, 10, 20, 30, 40, -1]:
        print(i, t[i])
        _s = s[i][0, data['res_in_boundary_layer']]
        _s = np.max(s[i][0], axis=0)
        ax.loglog(k[_s > 1e-20], _s[_s > 1e-20], label=f't={t[i]:.1f}')
    ax.legend(frameon=False)


def compare_spectra(Ra=1e8):
    fig, ax = plt.subplots()

    runs = []
    labels = []
    if Ra == 1e8:
        runs.append(get_pySDC_data('1e8', res=96, dt=0.005, config_name='RBC3DG4R4'))
        labels.append(r'$N=384^2\times96$')
        runs.append(get_pySDC_data('1e8', res=128, dt=0.005, config_name='RBC3DG4R4'))
        labels.append(r'$N=512^2\times128$')
        runs.append(get_pySDC_data('1e8', res=96, dt=0.006, config_name='RBC3DG4'))
        labels.append(r'$N=192^2\times96$')
        runs.append(get_pySDC_data('1e8', res=256, dt=0.002, config_name='RBC3DG4R4RK'))
        labels.append(r'$N=1024^2\times256$')
    elif Ra == 1e7:
        runs.append(get_pySDC_data('1e7', res=64, dt=0.005, config_name='RBC3DG4R4'))
        labels.append(r'$N=264^2\times64$')
        runs.append(get_pySDC_data('1e7', res=64, dt=0.005, config_name='RBC3DG4R4D2'))
        labels.append(r'$N=264^2\times64$, no dealiasing')
        runs.append(get_pySDC_data('1e7', res=64, dt=0.005, config_name='RBC3DG4R4D4'))
        labels.append(r'$N=264^2\times64$, dealiasing 2')
        runs.append(get_pySDC_data('1e7', res=96, dt=0.009, config_name='RBC3DG4'))
        labels.append(r'$N=192^2\times96$')
        runs.append(get_pySDC_data('1e7', res=96, dt=0.005, config_name='RBC3DG4R4'))
        labels.append(r'$N=384^2\times96$')
        runs.append(get_pySDC_data('1e7', res=128, dt=0.005, config_name='RBC3DG4R4'))
        labels.append(r'$N=512^2\times128$')
    elif Ra == 1e5:
        runs.append(get_pySDC_data('1e5', res=32, dt=0.06, config_name='RBC3DG4'))
        labels.append(r'$N=64^2\times32$')
        runs.append(get_pySDC_data('1e5', res=32, dt=0.06, config_name='RBC3DG4R4'))
        labels.append(r'$N=128^2\times32$')
    else:
        raise NotImplementedError

    for data, label in zip(runs, labels):
        s_all = data['spectrum']
        k = data['k']

        from pySDC.helpers.spectral_helper import ChebychevHelper

        helper = ChebychevHelper(N=s_all[0].shape[1], x0=0, x1=1)
        weights = helper.get_integration_weights()
        s = np.array([weights @ helper.transform(me[0], axes=(0,)) for me in s_all]).mean(axis=0)

        ax.loglog(k[s > 1e-20], s[s > 1e-20], label=label)

        # spectrum = np.array(data['spectrum'])
        # u_spectrum = np.mean(spectrum, axis=0)[1]
        # idx = data['res_in_boundary_layer']
        # _s = u_spectrum[idx]
        # ax.loglog(k[_s > 1e-20], _s[_s > 1e-20], label=label)

    ax.legend(frameon=False)
    ax.set_xlabel('$k$')
    ax.set_ylabel(r'$\|\hat{u}_x\|$')
    ax.set_title(f'Ra = {Ra:.1e}')


if __name__ == '__main__':
    # plot_Ra_Nusselt_scaling()

    compare_Nusselt_over_time1e5()
    # compare_Nusselt_over_time1e6()
    # compare_Nusselt_over_time1e7()
    # compare_Nusselt_over_time1e8()
    # plot_thibaut_stuff()
    # plot_spectrum_over_time1e6R4()
    # compare_spectra(Ra=1e5)

    plt.show()
