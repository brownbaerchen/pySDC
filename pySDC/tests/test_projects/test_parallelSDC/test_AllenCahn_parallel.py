import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize("variant", ['sl_parallel', 'ml_parallel'])
def test_parallel_variants(variant):
    import os
    import subprocess

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cmd = (
        "mpirun -np 3 python -c \"from pySDC.projects.parallelSDC.AllenCahn_parallel import *; "
        f"run_variant(\'{variant}');\""
    )
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s' % (p.returncode)


@pytest.mark.base
@pytest.mark.parametrize("variant", ['sl_serial', 'ml_serial'])
def test_serial_variants(variant):
    from pySDC.projects.parallelSDC.AllenCahn_parallel import run_variant

    run_variant(variant=variant)
