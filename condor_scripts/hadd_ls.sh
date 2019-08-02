#!/bin/bash
source /afs/cern.ch/user/s/sshirobo/FairShipRun/config.sh
set -ux
#folders=($(ls $2))
#no need for filename anymore
folders=(`find "$2" -name *.root`)
hadd "$1" $(eval echo ${folders[@]}) && xrdcp "$1" "$2"/"$1"
