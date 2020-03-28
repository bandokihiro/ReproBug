#!/bin/bash

set -e

LEGION_ROOT="/nobackupp2/kbando/MyLibs/legion_master_release"
METIS_ROOT="/nobackupp2/kbando/MyLibs/metis-5.1.0"
HDF5_ROOT="/nobackupp2/kbando/MyLibs/hdf5-1.10.6"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DMETIS_ROOT="$METIS_ROOT" \
    -DLEGION_ROOT="$LEGION_ROOT" \
    -DHDF5_ROOT="$HDF5_ROOT" \
    ..
