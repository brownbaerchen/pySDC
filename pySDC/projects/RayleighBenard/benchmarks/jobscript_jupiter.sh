#!/bin/bash -x
#SBATCH -n #PROCS#
#SBATCH --tasks-per-node=#PROCS_PER_NODE#
#SBATCH -p #PARTITION#
#SBATCH --time=#WALLTIME#
#SBATCH -e #ERROR_FILEPATH#
#SBATCH -o #OUT_FILEPATH#
#SBATCH --job-name=benchmark 
#SBATCH -A jureap1
#SBATCH --reservation=campaign
#SBATCH --dependency=singleton

### start of jobscript

source /e/project1/cjsc/baumann7/pySDC/pySDC/projects/GPU/etc/venv_jupiter/activate.sh

#EXEC#
touch #READY#
