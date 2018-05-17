#!/bin/bash
set -e

# Usage:
#   bash
#   source activate equinox
#   ./launch-jlab.sh 

source activate equinox

# delete existing jlab log file
JLAB_LOG='jlab.log'
rm -f $JLAB_LOG

echo "Launching job ..."
s=`qsub jlab.pbs`
# in order to have live log output, use instead:
#s=`qsub -k oe jlab.pbs`
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

