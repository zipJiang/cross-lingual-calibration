#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

BASE_DIR=$(pwd)/
LANG=
TASK=
COMMAND=
SERIALIZATION_DIR=

CUDA_DEVICES=0

while [[ $# -gt 0 ]];
do
    key="$1"
    case $key in
        --lang)
        LANG="$2"
        shift
        shift
        ;;
        --task)
        TASK="$2"
        shift
        shift
        ;;
        --serialization_dir)
        SERIALIZATION_DIR="$2"
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
DATA_DIR=${DATA_DIR}data/

export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"


# normalize serialization_dir
[[ "${SERIALIZATION_DIR}" != */ ]] && SERIALIZATION_DIR="${SERIALIZATION_DIR}/"

# make a temperary file that contains the jsonl
input_file_path=$(mktemp)
python3 ${SCRIPT_DIR}generate_evaluation_paths.py --lang ${LANG} --task ${TASK} --return_dict > ${input_file_path}

# resolve all directory issues
mkdir -p "${SERIALIZATION_DIR}calibration/"

# allennlp predict \
#     ${SERIALIZATION_DIR} \
#     ${input_file_path} \
#     --file-friendly-logging \
#     --include-package enc_pred \
#     --predictor span-label-predictor \
#     --cuda-device ${CUDA_DEVICES} \
#     --output-file "${DATA_DIR}calibration-${TASK}/${LANG}.jsonl"
python3 ${SCRIPT_DIR}predict_original_logits.py --archive_path ${SERIALIZATION_DIR} \
    --input_path ${input_file_path} \
    --cuda_device ${CUDA_DEVICES} \
    --output_path "${SERIALIZATION_DIR}calibration/${LANG}.jsonl"

rm ${input_file_path}
