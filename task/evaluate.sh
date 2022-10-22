#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1
export CUDA_DEVICES=0
export BATCH_SIZE=16
export NUM_WORKERS=0
export LEARNING_RATE=0.0001


BASE_DIR=$(pwd)/
SERIALIZATION_DIR=
TASK=
LANG=

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -s|--serialization_dir)
            SERIALIZATION_DIR="$2"
            [[ "${SERIALIZATION_DIR}" != */ ]] && SERIALIZATION_DIR="${SERIALIZATION_DIR}/"
            shift
            shift
            ;;
        -t|--task)
            TASK="$2"
            shift
            shift
            ;;
        -l|--lang)
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

SCRIPT_DIR="${BASE_DIR}scripts/"


# running evaluation on dataset
cd "${BASE_DIR}"
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"
export TASK
export LANG
export SERIALIZATION_DIR
export OUTPUT_FILENAME="${SERIALIZATION_DIR}eval/${LANG}.json"

mkdir -p ${SERIALIZATION_DIR}eval/


allennlp evaluate \
    "${SERIALIZATION_DIR}" \
    "$(python3 ${SCRIPT_DIR}generate_evaluation_paths.py --lang ${LANG} --task ${TASK})" \
    --include-package enc_pred \
    --file-friendly-logging \
    --output-file ${OUTPUT_FILENAME} \
    --cuda-device 0