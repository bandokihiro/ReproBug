#!/bin/bash

set -e

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DMETIS_ROOT=/home/kihiro/Softwares/metis-5.1.0 \
    -DLEGION_ROOT=/home/kihiro/Softwares/legion_master_release \
    -DHDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial \
    ..
