#!/bin/bash
source /cvmfs/ship.cern.ch/develop/setUp.sh
source /afs/cern.ch/user/p/pdebryas/private/aliconfig.sh
#set -ux
echo "Starting script."

ClusterId=$1
ProcId=$2
NJOBS=500
NTOTAL=200000

LSB_JOBINDEX=$((ProcId+1))
OUTPUT_PATH=/afs/cern.ch/work/p/pdebryas/

SEED=$(( RANDOM ))
N=$(( NTOTAL/NJOBS ))
FIRST=$(((NTOTAL/NJOBS)*(LSB_JOBINDEX-1)))

python "$FAIRSHIP"/macro/run_simScript.py --PG --pID 11 --Estart 200 --Eend 400 -n $N --firstEvent $FIRST --seed $SEED
mkdir "$OUTPUT_PATH"/$ProcId
mv ship.conical.PG_11-TGeant4.root "$OUTPUT_PATH"/$ProcId/ship.conical.PG_11-TGeant4_"$ProcId".root
rm geofile_full.conical.PG_11-TGeant4.root
rm ship.params.conical.PG_11-TGeant4.root
