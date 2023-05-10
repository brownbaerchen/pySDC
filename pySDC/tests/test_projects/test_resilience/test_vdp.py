import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_procs', [1, 2, 5, 8])
def test_main(num_procs):
    import pySDC.projects.Resilience.vdp as vdp
    import os
    import subprocess

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    # run code with different number of MPI processes
    cmd = f"mpirun -np {num_procs} python {vdp.__file__}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )


if __name__ == "__main__":
    test_main(1)
