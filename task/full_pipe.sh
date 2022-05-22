#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

# Running the full pipeline from model training to evaluation.
LOG_STEM=
PRETRAINED_MODEL=
STEP=0
DCONFIG_FILENAME=en-en.json
LEARNING_RATE=0.00001

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
TASK_DIR=${BASE_DIR}task/
RUN_DIR=${BASE_DIR}runs/

# 02/23/2020 --- Adding an optional task-list option, so that we don't have
# to always rerun the whole pipeline over all tasks.

declare -a TASK_LIST="wikiann xnli udparse"
CALIBRATION_STEP=0

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --stem)
        LOG_STEM="$2"
        shift
        shift
        ;;
        --step)
        STEP="$2"
        shift
        shift
        ;;
        -p|--pretrained)
        PRETRAINED_MODEL="$2"
        shift
        shift
        ;;
        --dconfig)
        DCONFIG_FILENAME="$2"
        shift
        shift
        ;;
        --task)
        TASK_LIST="$2"
        shift
        shift
        ;;
        --calibration-step)
        CALIBRATION_STEP="$2"
        shift
        shift
        ;;
        --learning_rate)
        LEARNING_RATE="$2"
        shift
        shift
    esac
done

export LOG_STEM

# UPDATE 02/24/2022
# try to submit jobs 

if [[ $STEP -le 0 ]]; then
    # Training models with the given stem and pretrained models
    ${TASK_DIR}train_all.sh \
        --pretrained "${PRETRAINED_MODEL}" \
        --stem "${LOG_STEM}" \
        --dconfig ${DCONFIG_FILENAME} \
        --task "${TASK_LIST}" \
        --learning_rate "${LEARNING_RATE}"
fi

if [[ $STEP -le 1 ]]; then
    # evaluate the original model and get performance metrics and etc.
    ${TASK_DIR}evaluate_all.sh \
        --stem "${LOG_STEM}" \
        --task "${TASK_LIST}"
fi

if [[ $STEP -le 2 ]]; then
    # inference and get logits for all tasks
    ${TASK_DIR}predict_all.sh \
        --stem "${LOG_STEM}" \
        --task "${TASK_LIST}"

fi

if [[ $STEP -le 3 ]]; then
    # calibrate all the models and evaluate.
    ${TASK_DIR}calibrate_all.sh \
        --stem "${LOG_STEM}" \
        --task "${TASK_LIST}" \
        --step ${CALIBRATION_STEP}
fi


# comment this out so that we no-longer pushes.
# if [[ $STEP -le 4 ]]; then
#     # push result to related sheet.
#     ${TASK_DIR}push_result.sh \
#         --stem "${LOG_STEM}"
# fi
