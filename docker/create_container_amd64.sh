#!/usr/bin/env bash

### EDIT THIS TO WHEREVER YOU'RE STORING YOU DATA ###
# folder should exist before you mount it
LOCAL_DATA_FOLDER=/home/jesse/Code/src/mrl-vrpn-client/recordings
LOCAL_RESULTS_FOLDER=~/Code/src/dynosam/results/
LOCAL_DYNO_SAM_FOLDER=~/Code/src/dynosam/DynOSAM
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=~/Code/src/dynosam/extras/

bash create_container_base.sh acfr_rpg/dyno_sam_cuda dyno_sam $LOCAL_DATA_FOLDER $LOCAL_RESULTS_FOLDER $LOCAL_DYNO_SAM_FOLDER $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER
