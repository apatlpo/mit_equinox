#!/bin/bash
set -e

# Usage:
#   bash
#   source activate pangeo
#   ./launch-dask-nmpi.sh [nworkers] [nmpi] [nthreads] [memory-limit]
#   ./launch-dask-nmpi.sh 4 2 4 1e10

source activate pangeo

# create a directory to store temporary dask data
mkdir -p $DATAWORK/dask 
echo "Clean up ${DATAWORK}/dask"

SCHEDULER=$DATAWORK/dask/scheduler.json
rm -f $SCHEDULER

#rm -rf $DATAWORK/dask/worker-*

nmpi=${2}
nthreads=${3}
ncpus=$(($nmpi * $nthreads))
mem=${4}
edit_config () {
sed -e "s/mpiprocs=7/mpiprocs=$nmpi/g" $1 > tmp0.sh
sed -e "s/--np 7/--np $nmpi/g" tmp0.sh > tmp1.sh
sed -e "s/ompthreads=7/ompthreads=$nmpi/g" tmp1.sh > tmp2.sh
sed -e "s/--nthreads 4/--nthreads $nthreads/g" tmp2.sh > tmp3.sh
sed -e "s/ncpus=28/ncpus=$ncpus/g" tmp3.sh > tmp4.sh
sed -e "s/--memory-limit 1e10/--memory-limit $mem/g" tmp4.sh > tmp5.sh
mv tmp5.sh $2
rm tmp*.sh
}


echo "Launching dask scheduler"
edit_config launch-dask-scheduler.sh scheduler-tmp.sh
s=`qsub scheduler-tmp.sh`
sjob=${s%.*}
echo ${s}

workers=${1:-4}
echo "Launching dask workers (${workers})"
edit_config launch-dask-worker.sh worker-tmp.sh
for i in $(seq 1 ${workers}); do
    qsub worker-tmp.sh
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


