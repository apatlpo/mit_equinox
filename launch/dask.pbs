#!/bin/bash
#PBS -N dask_pbs
#PBS -q mpi_8
#PBS -l select=8:ncpus=28:mem=100g
#PBS -l walltime=24:00:00

# Qsub template for datarmor
# Scheduler: PBS

# This writes a scheduler.json file into your home directory
# You can then connect with the following Python code
# >>> from dask.distributed import Client
# >>> client = Client(scheduler_file='scheduler.json')

#Environment sourcing
ENV_SOURCE="source ~/.bashrc; export PATH=$HOME/.miniconda3/bin:$PATH; source activate equinox"

echo $PBS_O_WORKDIR  # dir where pbs script was submitted
#SCHEDULER="$PBS_O_WORKDIR/scheduler.json"
SCHEDULER="$DATAWORK/dask/scheduler.json"
echo $SCHEDULER
rm -f $SCHEDULER

#Options
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo $NCPUS 
# can't have less than 28 cpus
NCPUS=14
MEMORY_LIMIT="100e9"
INTERFACE="--interface ib0 "

# Run Dask Scheduler
echo "*** Launching Dask Scheduler ***"
pbsdsh -n 0 -- /bin/bash -c "$ENV_SOURCE; dask-scheduler $INTERFACE --scheduler-file $SCHEDULER > $PBS_O_WORKDIR/$PBS_JOBID-scheduler-$PBS_TASKNUM.log 2>&1;"&

#Number of chunks
nbNodes=`cat $PBS_NODEFILE | wc -l`

echo "*** Starting Workers on Other $nbNodes Nodes ***"
for ((i=1; i<$nbNodes; i+=1)); do
    pbsdsh -n ${i} -- /bin/bash -c "$ENV_SOURCE; dask-worker $INTERFACE --scheduler-file $SCHEDULER --nthreads $NCPUS --memory-limit $MEMORY_LIMIT --local-directory $TMPDIR --name worker-${i};"&
    #pbsdsh -n ${i} -- /bin/bash -c "$ENV_SOURCE; dask-worker $INTERFACE --scheduler-file $PBS_O_WORKDIR/scheduler.json --nthreads $NCPUS --memory-limit $MEMORY_LIMIT --local-directory $TMPDIR --name worker-${i};"&
done

SNODE = $(cat $PBS_NODEFILE | uniq | head )
#echo "scheduler node should be: $SNODE" >  "$PBS_O_WORKDIR/$PBS_JOBID.scheduler"
cat $PBS_NODEFILE | uniq >  "$PBS_O_WORKDIR/$PBS_JOBID.nodefile"


echo "*** Dask cluster is starting ***"
#Either sleep or wait if just startin a cluster
#Or lanuch a dask app here
sleep 86400
