#!/bin/bash
set -e

# Usage:
#   bash
#   source activate equinox
#   ./launch-dask.sh 4 

source activate equinox

# create a directory to store temporary dask data
#mkdir -p $DATAWORK/dask 
#rm -rf $DATAWORK/dask/*
#
#mkdir -p $SCRATCH/dask 
#echo "Clean up ${SCRATCH}/dask"
#rm -rf $SCRATCH/dask/*


SCHEDULER=$DATAWORK/dask/scheduler.json
#SCHEDULER=$SCRATCH/dask/scheduler.json
rm -f $SCHEDULER

echo "Launching dask scheduler"
s=`qsub launch-dask-scheduler.sh`
sjob=${s%.*}
echo ${s}

workers=${1:-4}
echo "Launching dask workers (${workers})"
for i in $(seq 1 ${workers}); do
    qsub launch-dask-worker.sh
done

qstat ${sjob}

# block until the scheduler job starts
while true; do
    status=`qstat ${sjob} | tail -n 1`
    echo ${status}
    if [[ ${status} =~ " R " ]]; then
        break
    fi
    sleep 1
done

# wait for scheduler.json
while true; do
    if [ -f $SCHEDULER ]; then
        echo " scheduler.json has been found "
        break
    fi
    sleep 1
done

default=$HOME
notebook=${2:-$default}
echo "Setting up Jupyter Lab, Notebook dir: ${notebook}"
./setup-jlab.py --log_level=DEBUG --jlab_port=8877 --dash_port=8878 \
    --notebook_dir $notebook --scheduler_file $SCHEDULER


