#!/bin/bash
# This script is used to generate spanfinder related calibration files.

set -i
set -x

BASE_DIR=/brtx/604-nvme2/zpjiang/encode_predict/
SPANFINDER_BASE_DIR=/brtx/604-nvme2/zpjiang/spanfinder_eval/
SPANFINDER_CALIBRATION_FILE_DIR=${BASE_DIR}data/ace+better_tagging_typing/
TASK_DIR="${BASE_DIR}task/"

declare -a MODEL_LIST=("xlmr")
declare -a TASK_LIST=("ace")
declare -a FILE_STEM=("train" "dev" "test")

for model_name in "${MODEL_LIST[@]}"; do
    for task_name in "${TASK_LIST[@]}"; do
        # We first generate a directory to load prediction result.
        mkdir -p "${SPANFINDER_CALIBRATION_FILE_DIR}${model_name}_${task_name}_tagging"
        mkdir -p "${SPANFINDER_CALIBRATION_FILE_DIR}${model_name}_${task_name}_typing"

        for file_stem in "${FILE_STEM[@]}"; do
            ${TASK_DIR}predict_with_spanfinder.sh --span \
                --input_path ${SPANFINDER_CALIBRATION_FILE_DIR}${task_name}/${file_stem}.force.jsonl \
                --output_path "${SPANFINDER_CALIBRATION_FILE_DIR}${model_name}_${task_name}_tagging/calibration-${file_stem}.jsonl" \
                --archive_dir "${SPANFINDER_BASE_DIR}runs/${model_name}_${task_name}"
            ${TASK_DIR}predict_with_spanfinder.sh \
                --input_path ${SPANFINDER_CALIBRATION_FILE_DIR}${task_name}/${file_stem}.force.jsonl \
                --output_path "${SPANFINDER_CALIBRATION_FILE_DIR}${model_name}_${task_name}_typing/calibration-${file_stem}.jsonl" \
                --archive_dir "${SPANFINDER_BASE_DIR}runs/${model_name}_${task_name}"
        done

        # after generating required data-file, we run the calibration step for the model
        ${TASK_DIR}calibrate_spanfinder_predictions.sh \
            --data_dir "${SPANFINDER_CALIBRATION_FILE_DIR}${model_name}_${task_name}_tagging"

        ${TASK_DIR}calibrate_spanfinder_predictions.sh \
            --data_dir "${SPANFINDER_CALIBRATION_FILE_DIR}${model_name}_${task_name}_typing"
    done
done