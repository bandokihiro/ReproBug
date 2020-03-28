#!/bin/bash

EXEC="../exec"

#export LEGION_BACKTRACE=1
#export LEGION_FREEZE_ON_ERROR=1

FLAGS="$FLAGS -lg:no_tracing"
#FLAGS="$FLAGS -dm:memoize"
FLAGS="$FLAGS -lg:partcheck"
#FLAGS="$FLAGS -ll:show_rsrv"
#FLAGS="$FLAGS -ll:dma 1"
#FLAGS="$FLAGS -ll:util 1"
#FLAGS="$FLAGS -level legion_spy=2 -logfile log_%.spy" # for checking the runtime analysis, need -DLEGION_SPY
#FLAGS="$FLAGS -ll:force_kthreads"
FLAGS="$FLAGS -lg:inorder"

#NCPU_PER_RANK=3
#NRANK=8
#NRANK_PER_NODE=2

NCPU_PER_RANK=4
NRANK=8
NRANK_PER_NODE=8

mkdir -p logs
export MPI_DSM_DISTRIBUTE=0
export GASNET_ODP_VERBOSE=0
mpiexec -np $NRANK -ppn $NRANK_PER_NODE $EXEC $FLAGS -ll:cpu $NCPU_PER_RANK |& tee logs/log.out
