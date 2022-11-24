#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=01:00:00
#SBATCH --output=data/text_output/out%j.txt
#SBATCH -e=data/text_output/err%j.txt
#SBATCH -A cstma
#SBATCH --mail-type=ALL
#SBATCH -p batch

cd /p/project/ccstma/baumann7/pySDC/pySDC/projects/Resilience

source /p/project/ccstma/baumann7/miniconda/bin/activate pySDC

mpirun -n 80 python timings.py adaptivity True smooth False
mpirun -n 80 python timings.py adaptivity True smooth True
