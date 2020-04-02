#!/bin/bash

set -e
LEGION_ROOT=/home/steven/Documents/Research/legion/release
METIS_ROOT=/home/steven/Documents/Programs/metis-5.1.0
HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial
BUILD_TYPE=Release

cmake \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DMETIS_ROOT=$METIS_ROOT \
    -DLEGION_ROOT=$LEGION_ROOT \
    -DHDF5_ROOT=$HDF5_ROOT \
    ..
