#!/usr/bin/env bash
DOCKERFILE=$1
TAG=$2

if [[ -z $TAG ]] || [[ -z $DOCKERFILE ]]; then
    SCRIPT_NAME=$(basename "$0")
    echo "Usage: ./$SCRIPT_NAME DOCKERFILE TAG"
    exit -1
fi


# 2. Identify the directory where this script is located
#    This resolves the path even if called from a different location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 3. Temporarily change into the script's directory
echo "Changing working directory to: $SCRIPT_DIR"
cd "$SCRIPT_DIR" || { echo "Failed to change directory"; exit 1; }

# 4. Verify the file exists in this directory
if [ -f "$DOCKERFILE" ]; then
    echo "Building dockerfile $DOCKERFILE with tag $TAG"
    DOCKER_BUILDKIT=1 docker build --network=host -f $DOCKERFILE -t $TAG .
    # ----------------------------

else
    echo "Error: File '$DOCKERFILE' not found in $SCRIPT_DIR"
    exit 1
fi
