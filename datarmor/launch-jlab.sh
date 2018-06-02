#!/bin/bash
set -e

# Usage:
#   bash
#   source activate equinox
#   ./launch-jlab.sh   (does not try connect to dashboard)
#   ./launch-jlab.sh  wait  (wait for node file)
#   ./launch-jlab.sh  1208493.datarmor0.nodefile  (use a specific node file)

source activate equinox

# create a log file with random name and delete existing jlab log file if any
JLAB_LOG="jlab.$RANDOM.log"
rm -f $JLAB_LOG > /dev/null 2>&1

echo "Launching job ..."
#s=`qsub jlab.pbs`
if [ "$#" -eq 0 ]; then
    echo "no dask dashboard connection"
    DASHINFO="0"
else
    DASHINFO=$1
fi
s=`qsub -v JLAB_LOG=$JLAB_LOG,DASHINFO=$DASHINFO jlab.pbs`
# in order to have live log output, use the following qsub option: -k oe

sjob=${s%.*}
echo " ... ${s} launched"

# wait for jlab.log
while true; do
    if [ -f $JLAB_LOG ]; then
        echo "$JLAB_LOG has been found "
	cat ${JLAB_LOG}
        echo "Kill jlab job with:"
        echo "qdel ${sjob}"
        break
    fi
    sleep 1
done

