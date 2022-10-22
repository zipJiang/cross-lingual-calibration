#!/bin/bash


set -i
set -x

BASE_DIR=$(pwd)/
export PYTHONPATH=$(BASE_DIR):${PYTHONPATH}

TASK=
LANG=

while [[ $# -gt 0 ]];
do
    key="$1"
    case $key in
        -s|--serialization_dir)
            [[ "${SERIALIZATION_DIR}" != */ ]] && SERIALIZATION_DIR="${SERIALIZATION_DIR}/"
            SERIALIZATION_DIR="$2"
            shift
            shift
            ;;
        --task)
            TASK="$2"
            shift
            shift
            ;;
        --lang)
            LANG="$2"
            shift
            shift
            ;;
        --base_dir)
            BASE_DIR="$2"
            [[ "${BASE_DIR}" != */ ]] && BASE_DIR="${BASE_DIR}/"
            shift
            shift
            ;;
    esac
done

SCRIPT_DIR=${BASE_DIR}scripts/

INPUT_FILE_PATH=$(mktemp)

python3 ${SCRIPT_DIR}generate_evaluation_paths --lang ${LANG} --task ${TASK} --return_dict > ${INPUT_FILE_PATH}

[ ! -d "${SERIALIZATION_DIR}calibration" ] && mkdir -p "${SERIALIZATION_DIR}calibration"

python3 ${SCRIPT_DIR}predict_span_probs.py \
    --data_config_path ${INPUT_FILE_PATH} \
    --device 0 \
    --output_path ${SERIALIZATION_DIR}calibration/${LANG}.jsonl \
    --archive_path ${SERIALIZATION_DIR}