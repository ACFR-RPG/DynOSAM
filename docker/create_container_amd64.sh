#!/usr/bin/env bash

### EDIT THIS TO WHEREVER YOU'RE STORING YOU DATA ###
# folder should exist before you mount it
LOCAL_DATA_FOLDER=~/logs_dynosam/
LOCAL_RESULTS_FOLDER=~/results_dynosam/
LOCAL_DYNO_SAM_FOLDER=~/git_dynosam/DynOSAM
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=~/git_dynosam/extras/

bash create_container_base.sh acfr_rpg/dyno_sam_cuda dyno_sam $LOCAL_DATA_FOLDER $LOCAL_RESULTS_FOLDER $LOCAL_DYNO_SAM_FOLDER $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER
