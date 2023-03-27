#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=00:15:00
#SBATCH --output=out/out%j.txt
#SBATCH --error=out/err%j.txt
#SBATCH -A cstma
#SBATCH --mail-type=ALL
#SBATCH -p gpus
#SBATCH -J GPU
#SBATCH --gres=gpu:1

module --force purge
module load Stages/2022
module load Intel/2021.4.0
module load IntelMPI/2021.4.0

cd /p/project/ccstma/baumann7/pySDC/pySDC/projects/Resilience

source /p/project/ccstma/baumann7/miniconda/bin/activate pySDC

srun -n 1 python ${1}
