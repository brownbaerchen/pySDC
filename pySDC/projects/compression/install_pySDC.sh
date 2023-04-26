#!/bin/bash

# store current working directory to return here later
current_dir=$(pwd)

# load libpressio in spack to make sure we are using the correct Python
spack load libpressio

# install local version of pySDC
cd /pySDC
pip install -e .
python -m pip install pytest

# go back to original working directory
cd $current_dir
