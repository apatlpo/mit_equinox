#!/bin/bash

# Usage:
#   ./archive run_name
#


DIR="run_$1"

echo "archives in $DIR"

mkdir $DIR

grep GMT parcel_distributed.pbs.o*

mv dask-worker.o* parcel_distributed.pbs.o* distributed.log dask-report-* dask-memory*.csv $DIR



