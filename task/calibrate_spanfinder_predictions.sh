#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

DAT_DIR=
TRAIN_DATA_PATH=
VALIDATION_DATA_PATH=
TEST_DATA_PATH=
LEARNING_RATE=0.01
LABEL_KEY='label'
LOGIT_KEY='logit'

export BATCH_SIZE=4096
export CONGIFURATION_PATH=/brtx/604-nvme2/zpjiang/encode_predict/configs/calibration.jsonnet

# CONTROL FLOW
STEP=0

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -l|--learning_rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        # --train_data_path)
        #     TRAIN_DATA_PATH="$2"
        #     shift
        #     shift
        #     ;;
        # --validation_data_path)
        #     VALIDATION_DATA_PATH="$2"
        #     shift
        #     shift
        #     ;;
        # --test_data_path)
        #     TEST_DATA_PATH="$2"
        #     shift
        #     shift
        #     ;;
        --data_dir)
            DATA_DIR="$2"
            [[ "${DATA_DIR}" != */ ]] && DATA_DIR="${DATA_DIR}/"
            shift
            shift
            ;;
        --logit_key)
            LOGIT_KEY="$2"
            shift
            shift
            ;;
        --label_key)
            LABEL_KEY="$2"
            shift
            shift
            ;;
        # -s|--serialization_dir)
        #     SERIALIZATION_DIR="$2"
        #     [[ "${SERIALIZATION_DIR}" != */ ]] && SERIALIZATION_DIR="${SERIALIZATION_DIR}/"
        #     shift
        #     shift
        #     ;;
        --step)
            STEP="$2"
            shift
            shift
            ;;
    esac
done

eval "$(conda shell.bash hook)"
conda activate /brtx/604-nvme1/zpjiang/spanfinder/.env
BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
SCRIPT_DIR="${BASE_DIR}scripts/"

export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"
export LEARNING_RATE
export LABEL_KEY
export LOGIT_KEY

# extract data_file_path from the data_dir
export DATA_DIR
export SERIALIZATION_DIR=${DATA_DIR}/calibration/
export TRAIN_DATA_PATH="${DATA_DIR}calibration-train.jsonl"
export VALIDATION_DATA_PATH="${DATA_DIR}calibration-dev.jsonl"
export TEST_DATA_PATH="${DATA_DIR}calibration-test.jsonl"


# TODO: adding a layer of step control

if [[ $STEP -le 0 ]]; then
    rm -rf "${SERIALIZATION_DIR}"
    allennlp train \
        --include-package enc_pred \
        --file-friendly-logging \
        -s "${SERIALIZATION_DIR}" \
        ${CONGIFURATION_PATH}
fi

if [[ $STEP -le 1 ]]; then
    mkdir -p ${SERIALIZATION_DIR}eval/
    for filename in $(ls ${DATA_DIR}*.jsonl); do
        filename=$(basename ${filename})
        # change suffix
        if [[ "${filename}" != calibration* ]]; then
            allennlp evaluate \
                ${SERIALIZATION_DIR} \
                "${DATA_DIR}${filename}" \
                --include-package enc_pred \
                --file-friendly-logging \
                --output-file "${SERIALIZATION_DIR}eval/${filename/jsonl/json}"
        fi
    done
fi