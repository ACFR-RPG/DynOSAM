### EDIT THIS TO WHEREVER YOU'RE STORING YOU DATA ###
# folder should exist before you mount it
LOCAL_DATA_FOLDER=/media/jmor6670/T7/datasets/
LOCAL_RESULTS_FOLDER=/home/usyd/dynosam/results/
LOCAL_DYNO_SAM_FOLDER=/home/usyd/dynosam/DynOSAM/
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=/home/usyd/dynosam/extra

bash create_container_base.sh acfr_rpg/dynosam_cuda_l4t $LOCAL_DATA_FOLDER $LOCAL_RESULTS_FOLDER $LOCAL_DYNO_SAM_FOLDER $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER