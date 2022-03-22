#!/bin/bash

set -i
set -x

# This shellscript run SpanFinder prediction over force-decoding input files
# and generate a set of input files for the calibration module.
eval "$(conda shell.bash hook)"
conda activate /brtx/604-nvme1/zpjiang/spanfinder/.env
BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
SPANFINDER_BASE_DIR="/brtx/604-nvme2/zpjiang/spanfinder_eval/"
SCRIPT_DIR="${BASE_DIR}scripts/"

export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"
export SPANFINDER_DATA_DIR="${BASE_DIR}data/ace+better/"

declare -A to_targ
to_targ[train]="calibration-train"
to_targ[dev]="calibration-dev"
to_targ[test]="ar"

# for task in ace better; do
    task=better
    for depth in d0 d1; do
        source_dir="${SPANFINDER_DATA_DIR}${task}-${depth}/"
        for model_type in mbert xlmr; do
            # iterate through filename in the source dir
            mkdir -p ${source_dir}${model_type}/
            for filepath in ${source_dir}*; do
                if [ -d "${filepath}" ]; then
                    continue
                fi

                filename=${filepath##*/}
                python3 ${SPANFINDER_BASE_DIR}scripts/predict_force.py \
                    --archive_path "${SPANFINDER_BASE_DIR}runs/${model_type}_${task}" \
                    --input_path "$filepath" \
                    --write_to "${source_dir}${model_type}/${to_targ[${filename%\.force\.jsonl}]}.jsonl" \
                    --cuda_device 0
            done
        done
    done
# done