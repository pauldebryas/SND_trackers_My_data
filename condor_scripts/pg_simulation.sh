#!/bin/bash
source /afs/cern.ch/user/s/sshirobo/FairShipRun/config.sh
set -ux
echo "Starting script."
DIR=$1
ProcId=$2
LSB_JOBINDEX=$((ProcId+1))
OUTPUT_PATH=/eos/experiment/ship/user/sshirobo/"$DIR"/"$LSB_JOBINDEX"
NTOTAL=500000
NJOBS=$3
TANK=6
ISHIP=3
SEED=$(( RANDOM ))
N=$(( NTOTAL/NJOBS + ( LSB_JOBINDEX == NJOBS ? NTOTAL % NJOBS : 0 ) ))
FIRST=$(((NTOTAL/NJOBS)*(LSB_JOBINDEX-1)))
if eos stat "$OUTPUT_PATH"/ship.conical.PG_11-TGeant4.root; then
    echo "Target exists, nothing to do."
    exit 0
else
    python2 "$FAIRSHIP"/macro/run_simScript.py --PG --Estart 1 --Eend 40 --pID 11 --nEvents $N --firstEvent $FIRST\
            --seed $SEED --tankDesign $TANK --nuTauTargetDesign $ISHIP
    xrdcp ship.conical.PG_11-TGeant4.root root://eospublic.cern.ch/"$OUTPUT_PATH"/ship.conical.PG_11-TGeant4.root
#    if [ "$LSB_JOBINDEX" -eq 1 ]; then
#        xrdcp geofile_full.conical.MuonBack-TGeant4.root root://eospublic.cern.ch//eos/experiment/ship/user/olantwin/"$DIR"/
#    fi
fi
