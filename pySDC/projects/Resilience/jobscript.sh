#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=02:20:00
#SBATCH --output=data/text_output/out%A_%a.txt
#SBATCH --error=data/text_output/err%A_%a.txt
#SBATCH -A cstma
#SBATCH --mail-type=ALL
#SBATCH -p batch
#SBATCH -J MSSDC_timings
#SBATCH --array=0-3

module --force purge
module load Stages/2022
module load Intel/2021.4.0
module load IntelMPI/2021.4.0

cd /p/project/ccstma/baumann7/pySDC/pySDC/projects/Resilience

source /p/project/ccstma/baumann7/miniconda/bin/activate pySDC

problems=(run_advection run_heat)
smooth=(True False)

srun -n 80 python timings.py problem ${problems[${SLURM_ARRAY_TASK_ID} % 2]} adaptivity True smooth ${smooth[(${SLURM_ARRAY_TASK_ID} / 2) % 2]}
