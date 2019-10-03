#!/bin/bash
# Usage:

# ./launch

rm daskf.pbs &> /dev/null

# read from command line the number of arguments used
if [ $# -eq 0 ]; then
    export NNODES=8
    cp dask.pbs daskf.pbs
else
    export NNODES=$1
    if (( $NNODES > 18 )); then
        export QUEUE="big" 
    else
        export QUEUE="mpi_$NNODES"
    fi
    sed -e "s/mpi_8/$QUEUE/g" dask.pbs > daskf.pbs 
    sed -i -e "s/select=8/select=$NNODES/g" daskf.pbs
fi

echo "Number of nodes used for dask: $NNODES"

qsub -m n daskf.pbs



