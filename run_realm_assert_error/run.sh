#!/bin/bash

EXEC="../exec"

#export LEGION_BACKTRACE=1
#export LEGION_FREEZE_ON_ERROR=1

FLAGS="$FLAGS -logfile logs/log_%.log"
#FLAGS="$FLAGS -hdf5:forcerw"
#FLAGS="$FLAGS -lg:no_tracing"
#FLAGS="$FLAGS -ll:util 1"
#FLAGS="$FLAGS -ll:dma 3"
#FLAGS="$FLAGS -dm:memoize"

#FLAGS="$FLAGS -lg:partcheck"
#FLAGS="$FLAGS -ll:show_rsrv"
#FLAGS="$FLAGS -ll:force_kthreads"
#FLAGS="$FLAGS -lg:inorder"
#FLAGS="$FLAGS -level legion_spy=2 -logfile log_%.spy" # for checking the runtime analysis, need -DLEGION_SPY

#FLAGS="$FLAGS -lg:prof 1 -lg:prof_logfile prof_%.gz"

NCPU_PER_RANK=8
NRANK=2
NRANK_PER_NODE=1

# preparation
mkdir -p logs

# specific to pleiades
#export MPI_DSM_DISTRIBUTE=0
#export GASNET_ODP_VERBOSE=0

CMD="mpiexec -np $NRANK -ppn $NRANK_PER_NODE $EXEC $FLAGS -ll:cpu $NCPU_PER_RANK"

echo $CMD
$CMD
#$CMD |& tee log.out
