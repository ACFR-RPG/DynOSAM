#!/usr/bin/env bash

### EDIT THIS TO WHEREVER YOU'RE STORING YOU DATA ###
# folder should exist before you mount it
LOCAL_DATA_FOLDER=/media/jmor6670/T7/datasets
LOCAL_RESULTS_FOLDER=~/results/
LOCAL_DYNO_SAM_FOLDER=~/Code/src/DynOSAM/
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=~/Code/src/third_party_dynosam/

bash create_container_base.sh acfr_rpg/dyno_sam_cuda dyno_sam $LOCAL_DATA_FOLDER $LOCAL_RESULTS_FOLDER $LOCAL_DYNO_SAM_FOLDER $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER
