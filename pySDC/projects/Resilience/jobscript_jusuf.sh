#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --output=data/text_output/out%j.txt
#SBATCH --error=data/text_output/err%j.txt
#SBATCH -A cstma
#SBATCH --mail-type=END
#SBATCH --mail-user=t.baumann@fz-juelich.de
#SBATCH -p batch
#SBATCH -J save_world

module --force purge
module load Stages/2022
module load Intel/2021.4.0
module load IntelMPI/2021.4.0

cd /p/project/ccstma/baumann7/pySDC/pySDC/projects/Resilience

source /p/project/ccstma/baumann7/miniconda/bin/activate pySDC

srun -n 64 python ${1}
