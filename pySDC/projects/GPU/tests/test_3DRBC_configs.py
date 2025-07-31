import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('res', [8, 16, 32])
def test_ics_iterative(tmpdir, res, ic_res=8):
    from pySDC.projects.GPU.configs.RBC3D_configs import RBC3DscalingIterative
    from pySDC.helpers.fieldsIO import FieldsIO
    from mpi4py import MPI

    FieldsIO.ALLOW_OVERWRITE = True

    comm = MPI.COMM_WORLD

    args = {
        'procs': [1, 1, comm.size],
        'mode': 'run',
        'o': tmpdir,
        'useGPU': False,
        'res': res,
    }

    config = RBC3DscalingIterative(args, comm_world=comm)
    config.ic_res = ic_res
    config.ic_time = 0

    desc = config.get_description(res=res)
    N_low = config.ic_res
    prob = desc['problem_class'](
        **{**desc['problem_params'], 'nx': N_low, 'ny': N_low, 'nz': N_low, 'spectral_space': False}
    )
    prob.setUpFieldsIO()
    xp = prob.xp

    def get_sol(prob):
        me = prob.u_init
        me[...] = xp.cos(2 * xp.pi * prob.X) * xp.sin(4 * xp.pi * prob.Y) * prob.Z**2
        if prob.spectral_space:
            res = prob.u_init_forward
            res[...] = prob.transform(me)
            return res
        else:
            return me

    # prepare ICs
    u0 = get_sol(prob)

    outfile = prob.getOutputFile(f'{tmpdir}/{type(config).__name__}-res{config.ic_res}-ic.pySDC')
    outfile.addField(0, prob.processSolutionForOutput(u0))
    del outfile
    del prob

    # get ICs
    prob = desc['problem_class'](**{**desc['problem_params'], 'spectral_space': True})
    prob.setUpFieldsIO()
    ics, _ = config.get_initial_condition(prob)

    # check that ICs are correct
    expect = get_sol(prob)
    ics_hat = prob.transform(ics)

    error = abs(ics - expect)
    assert error < 1e-12, f'Got {error=} when interpolating ics'
    assert xp.allclose(ics_hat.shape[1:], res)


@pytest.mark.mpi4py
@pytest.mark.parametrize('spectral_space', [True, False])
@pytest.mark.parametrize('res_fac', [1, 2])
@pytest.mark.mpi(ranks=[1, 2])
def test_ics_verification(mpi_ranks, tmpdir, spectral_space, res_fac):
    from pySDC.projects.GPU.configs.RBC3D_configs import RBC3Dverification
    from pySDC.helpers.fieldsIO import FieldsIO
    from mpi4py import MPI
    import os

    FieldsIO.ALLOW_OVERWRITE = True

    comm = MPI.COMM_WORLD
    tmpdir = comm.bcast(tmpdir)

    args = {
        'procs': [1, 1, comm.size],
        'mode': 'run',
        'o': tmpdir,
        'useGPU': False,
        'res': -1,
        'dt': -1,
    }

    class Low1e0(RBC3Dverification):
        res = 8
        base_path = tmpdir

    class High1e1(RBC3Dverification):
        res = Low1e0.res
        ic_config = Low1e0
        base_path = tmpdir

    def get_sol(prob):
        me = prob.u_init_physical
        me[...] = prob.Z**2 * xp.cos(2 * xp.pi * prob.X / prob.axes[0].L) * xp.sin(4 * xp.pi * prob.Y / prob.axes[1].L)
        if prob.spectral_space:
            res = prob.u_init_forward
            res[...] = prob.transform(me)
            return res
        else:
            return me

    # make sure the paths are in place
    os.makedirs(f'{tmpdir}/data', exist_ok=True)

    # prepare solution to be interpolated
    config = Low1e0(args, comm_world=comm)

    desc = config.get_description()
    prob = desc['problem_class'](**{**desc['problem_params'], 'spectral_space': spectral_space})
    prob.setUpFieldsIO()
    xp = prob.xp

    u0 = get_sol(prob)

    outfile = prob.getOutputFile(config.get_file_name())
    outfile.addField(0, prob.processSolutionForOutput(u0))

    # cleanup
    del outfile
    del prob
    del xp
    del config

    # interpolate solution
    config = High1e1(args, comm_world=comm)

    desc = config.get_description(res=Low1e0.res * res_fac)
    prob = desc['problem_class'](**{**desc['problem_params'], 'spectral_space': spectral_space})
    prob.setUpFieldsIO()
    xp = prob.xp

    u0, _ = config.get_initial_condition(prob)
    u0_expect = get_sol(prob)
    assert Low1e0.res * res_fac in u0.shape[1:]
    assert xp.allclose(u0, u0_expect)


if __name__ == '__main__':
    test_ics_verification(None, './', True, 2)
