#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=01:00:00
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

srun -n 80 python timings.py problem run_advection adaptivity True smooth False
srun -n 80 python timings.py problem run_advection adaptivity True smooth True
srun -n 80 python timings.py problem run_heat adaptivity True smooth False
srun -n 80 python timings.py problem run_heat adaptivity True smooth True
