#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

set -i
set -x

PRETRAINED_MODEL="xlm-roberta-base"
BASE_DIR=$(pwd)/
WITH_CRF=false
RECOVER_FLAG=

export CUDA_DEVICES=0
export NUM_WORKERS=0
export EPOCHS=128
export LEARNING_RATE=0.00001
export SEQTAG_TASK=pos
export PATIENCE=8

BATCH_SIZE=32


while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -s|--serialization_dir)
            SERIALIZATION_DIR="$2"
            shift
            shift
            ;;
        -c|--configuration)
            CONFIGURATION_PATH="$2"
            shift
            shift
            ;;
        -d|--data_config)
            # Notice that here the data config is a path
            DATA_CONFIG="$2"
            shift
            shift
            ;;
        --task)
            SEQTAG_TASK="$2"
            shift
            shift
            ;;
        -p|--pretrained)
            PRETRAINED_MODEL="$2"
            [[ ${PRETRAINED_MODEL} == *large* ]] && BATCH_SIZE=8
            echo "${PRETRAINED_MODEL}"
            echo "batch_size = ${BATCH_SIZE}"
            shift
            shift
            ;;
        -l|--learning_rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        --crf)
            WITH_CRF=true
            shift
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --patience)
            PATIENCE="$2"
            shift
            shift
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift
            shift
            ;;
        --recover)
            RECOVER_FLAG=--recover
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

# if [ -z "${RECOVER_FLAG}" ]; then
#     rm -rf "${SERIALIZATION_DIR}"
# fi

cd "${BASE_DIR}"
export PRETRAINED_MODEL
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"
export DATA_PATH="$(cat ${DATA_CONFIG})"
export EPOCHS
export PATIENCE
export BATCH_SIZE
export SEQTAG_TASK
export WITH_CRF

python3 -um allennlp train --include-package enc_pred --file-friendly-logging \
    ${RECOVER_FLAG} -s "${SERIALIZATION_DIR}" \
    ${CONFIGURATION_PATH}