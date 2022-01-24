#!/bin/bash

# Usage:
#   ./archive run_name
#


DIR="run_$1"

echo "archives in $DIR"

mkdir $DIR

grep GMT time_dependent_interpolation.pbs.o*

mv dask-worker.o* time_dependent_interpolation.pbs.o* distributed.log dask-report-* $DIR



