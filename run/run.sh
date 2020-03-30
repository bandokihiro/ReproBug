#!/bin/bash

EXEC="../exec"

#export LEGION_BACKTRACE=1
#export LEGION_FREEZE_ON_ERROR=1

FLAGS="$FLAGS -lg:no_tracing"
FLAGS="$FLAGS -lg:inorder"
FLAGS="$FLAGS -lg:partcheck"

NCPU_PER_RANK=4
NRANK=8
NRANK_PER_NODE=8

# specific for pleiades
#export MPI_DSM_DISTRIBUTE=0
#export GASNET_ODP_VERBOSE=0

mpiexec -np $NRANK -ppn $NRANK_PER_NODE $EXEC $FLAGS -ll:cpu $NCPU_PER_RANK |& tee logs/log.out
