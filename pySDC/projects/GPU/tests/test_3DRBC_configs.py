import pytest


@pytest.mark.mpi4py
def test_ics(tmpdir, res=16, ic_res=8):
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
        me[...] = 1  # xp.sin(2*xp.pi*prob.X) * xp.sin(4*xp.pi*prob.Y) * prob.Z
        return me

    # prepare ICs
    u0 = get_sol(prob)
    u0_hat = prob.transform(u0)

    outfile = prob.getOutputFile(f'{tmpdir}/{type(config).__name__}-res{config.ic_res}-ic.pySDC')
    outfile.addField(0, prob.processSolutionForOutput(u0))
    del outfile
    del prob

    # get ICs
    prob = desc['problem_class'](**{**desc['problem_params'], 'spectral_space': False})
    prob.setUpFieldsIO()
    ics, _ = config.get_initial_condition(prob)

    # check that ICs are correct
    expect = get_sol(prob)
    ics_hat = prob.transform(ics)

    error = abs(ics - expect)
    print(error)
    breakpoint()
    assert xp.allclose(ics, xp.sin(2 * xp.pi * prob.X) * xp.sin(4 * xp.pi * prob.Y) * prob.Z)


if __name__ == '__main__':
    test_ics('./data', ic_res=4, res=8)
