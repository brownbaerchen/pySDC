FROM spack/ubuntu-jammy

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update -y && apt install -y cmake ccache pkg-config python3 pip

RUN spack install libpressio+python+sz3

# install the dependencies of pySDC
# RUN spack load libpressio
# RUN eval `spack load --sh   libpressio`  # bash/zsh/sh
# RUN pip install pySDC
# RUN pip uninstall -y pySDC
