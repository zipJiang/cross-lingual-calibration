#!/bin/bash

# This script is used to run prediction with spanfinder
# on certain datafiles.
set -i
set -x

SPANFINDER_BASE=/brtx/604-nvme2/zpjiang/spanfinder_eval/
SPANFINDER_SCRIPT_DIR="${SPANFINDER_BASE}scripts/"
ARCHIVE_DIR="${SPANFINDER_BASE}runs/xlmr_ace/"
INPUT_PATH=
OUPTUT_PATH=
WITH_SPANS=false

eval "$(conda shell.bash hook)"
conda activate /brtx/604-nvme1/zpjiang/spanfinder/.env
BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
SCRIPT_DIR="${BASE_DIR}scripts/"

cd ${SPANFINDER_BASE}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --archive_dir)
            ARCHIVE_DIR="$2"
            shift
            shift
            ;;
        --input_path)
            INPUT_PATH="$2"
            shift
            shift
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift
            shift
            ;;
        --span)
            WITH_SPANS=true
            shift
            ;;
    esac
done

# do the prediction over items
span_flag=
[ ${WITH_SPANS} = true ] && span_flag="--predict_spans"

python3 ${SPANFINDER_SCRIPT_DIR}predict_force.py --input_path ${INPUT_PATH} \
    --write_to ${OUTPUT_PATH} \
    --archive_path ${ARCHIVE_DIR} \
    --cuda_device 0 \
    ${span_flag}