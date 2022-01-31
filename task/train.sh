#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1


export PRETRAINED_MODEL="xlm-roberta-base"
export CUDA_DEVICES=0
export BATCH_SIZE=16
export NUM_WORKERS=0
export LEARNING_RATE=0.0001

TASK="pos_tags"


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
            DATA_CONFIG="$2"
            shift
            shift
            ;;
        -t|--task)
            TASK="$2"
            shift
            shift
            ;;
    esac
done

eval "$(conda shell.bash hook)"
conda activate /brtx/604-nvme1/zpjiang/spanfinder/.env
BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"

rm -rf "${SERIALIZATION_DIR}"
cd "${BASE_DIR}"
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"
export DATA_PATH="$(cat ${DATA_CONFIG})"
export TASK

python3 -um allennlp train --include-package enc_pred --file-friendly-logging \
    -s "${SERIALIZATION_DIR}" \
    ${CONFIGURATION_PATH}