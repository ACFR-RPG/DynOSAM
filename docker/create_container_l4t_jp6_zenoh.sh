#!/usr/bin/env bash

CONTAINER_IMAGE_NAME=dynosam2_l4t
CONTAINER_NAME=dynosam2_l4t

LOCAL_DATA_FOLDER=/media/jmor6670/T7/datasets/
LOCAL_RESULTS_FOLDER=~/Code/src/dynosam/results/
LOCAL_DYNO_SAM_FOLDER=~/Code/src/dynosam/DynOSAM
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=~/Code/src/dynosam/extras/

echo "Creating dynosam container ($CONTAINER_NAME) from image: $CONTAINER_IMAGE_NAME"
echo "Local data folder: $LOCAL_DATA_FOLDER"
echo "Local results folder: $LOCAL_RESULTS_FOLDER"
echo "Local DynoSAM folder: $LOCAL_DYNO_SAM_FOLDER"
echo "Local third party dynosam folder: $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER"


CONTAINER_DATA_FOLDER=/root/data
CONTAINER_RESULTS_FOLDER=/root/results
CONTAINER_WORKSPACE_FOLDER=/home/user/dev_ws/src/core/
CONTAINER_WORKSPACE_FOLDER_THIRD_PARTY=/home/user/dev_ws/src/third_parties/


USE_NVIDIA=false

# If we are running in a docker-in-docker scenario NVIDIA_SOS will be populated
# and we dont need to check for gpu or roscore
if [ -z "$NVIDIA_SOS" ]; then
    # Is there an nvidia card in the system?
    if [[ $(lshw -C display 2> /dev/null | grep -i NVIDIA) ]]; then
        echo "Nvidia GPU in the system, trying to use it."
        # If we have nvidia-container-runtime...
        if nvidia-container-runtime -v > /dev/null; then
            # Check if we have an nvidia driver
            NVIDIA_DRIVER_VERSION=`nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk '{ print substr($0, 0, 4) }'`
            if [ -z "$NVIDIA_DRIVER_VERSION" ]; then
                echo "No Nvidia driver version found, but nvidia-container-runtime is installed... can you run 'nvidia-smi?"
                echo "You may want to install a driver to take advantage of it."
            else
                USE_NVIDIA=true
            fi
        else
            echo "Warning:"
            echo "The system has a Nvidia GPU but doesn't have nvidia-container-runtime installed"
            echo "You may want to install it to take advantage of it, e.g.:"
            echo "sudo apt-get install nvidia-container-runtime"
        fi
        if ! "$USE_NVIDIA"; then
            echo "Using non-nvidia alternative."
        fi
    fi
else
    USE_NVIDIA=true
    DOCKER_IN_DOCKER=true
fi


XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

echo "Mounting data folder (local) $LOCAL_DATA_FOLDER -> (container) $CONTAINER_DATA_FOLDER"


if [ -z "$NVIDIA_SOS" ]; then
    # Compose the long list of .so files from Nvidia to make X work with seawolf
    # This provides a list of all .so files installed from packages from 'nvidia-''
    NVIDIA_SOS=$(dpkg -l | grep nvidia- | awk '{print $2}' | xargs dpkg-query -L | grep lib | grep .so)
    # This generates --volume /path/to/.so:/path/to/.so lines for every .so
fi
for so in $NVIDIA_SOS; do DOCKER_NVIDIA_SO_VOLUMES+="--volume $so:$so "; done


USE_NVIDIA=true

if "$USE_NVIDIA"; then
    # If executing this script from an unmanned shell (like from seawolf ShellCmd)
    # We can't use "-i"
    if [ -t 1 ]; then
        TERMINAL_FLAGS='-it'
    else
        TERMINAL_FLAGS='-t'
    fi

    echo "$@"
    # Create the container based on the launchfile it's launching (if any)
    # removes '.launch' from the last argument
    echo "Container name will be: $CONTAINER_NAME"
    docker run \
        --privileged \
        --gpus all \
        -i -d \
        --shm-size=512m \
        --volume $XSOCK:$XSOCK:rw \
        -v $LOCAL_DATA_FOLDER:$CONTAINER_DATA_FOLDER \
        -v $LOCAL_RESULTS_FOLDER:$CONTAINER_RESULTS_FOLDER \
        -v $LOCAL_DYNO_SAM_FOLDER:$CONTAINER_WORKSPACE_FOLDER \
        -v $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER:$CONTAINER_WORKSPACE_FOLDER_THIRD_PARTY \
        -v /var/run/docker.sock:/var/run/docker.sock \
        --env ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/link/tx/queue/congestion_control/drop/wait_before_drop=1000000' \
        --env ZENOH_ROUTER_CONFIG_URI="/root/.config/zenoh_config_local.json5" \
        --env DISPLAY=$DISPLAY \
        --env XAUTHORITY=$XAUTH \
        --env QT_X11_NO_MITSHM=0 \
        --env QT_X11_NO_XRENDER=0 \
        --volume $XAUTH:$XAUTH:rw \
        --device=/dev/bus/usb:/dev/bus/usb \
        -v /dev:/dev \
        -v /run/udev:/run/udev:ro \
        --net host \
        --pid host \
        --ipc host \
        -it \
         --name=$CONTAINER_NAME \
        --entrypoint /bin/bash \
        $CONTAINER_IMAGE_NAME \
        -lc '
            set -e
            source "/opt/ros/${ROS_DISTRO}/setup.bash"
            ros2 run rmw_zenoh_cpp rmw_zenohd &
            exec /bin/bash
        '
fi

# FOR NOW?
# xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $CONTAINER_NAME`
