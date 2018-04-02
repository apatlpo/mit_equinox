#!/bin/bash
set -e

# Usage:
#   bash
#   source activate pangeo
#   ./launch-jlab.sh 

source activate pangeon

# delete existing jlab log file
JLAB_LOG='jlab.log'
rm -f $JLAB_LOG

echo "Launching job ..."
s=`qsub jlab.pbs`
sjob=${s%.*}
echo " ... ${s} done"

#qstat -f ${sjob}

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

