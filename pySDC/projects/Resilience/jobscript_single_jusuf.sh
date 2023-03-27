#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=02:00:00
#SBATCH --output=data/text_output/out%j.txt
#SBATCH --error=data/text_output/err%j.txt
#SBATCH -A cstma
#SBATCH --mail-type=ALL
#SBATCH -p batch
#SBATCH -J MSSDC_timings

module --force purge
module load Stages/2022
module load Intel/2021.4.0
module load IntelMPI/2021.4.0

cd /p/project/ccstma/baumann7/pySDC/pySDC/projects/Resilience

source /p/project/ccstma/baumann7/miniconda/bin/activate pySDC

problems=(run_advection run_heat)
smooth=(True False)

srun -n 12 python timings.py problem run_vdp precon IE adaptivity True estimate_semi_glob_error True
srun -n 12 python timings.py problem run_vdp precon IE adaptivity True estimate_semi_glob_error False
srun -n 12 python timings.py problem run_vdp precon LU adaptivity True estimate_semi_glob_error False
