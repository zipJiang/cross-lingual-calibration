# !/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

# Running the full pipeline from model training to evaluation.
LOG_STEM=
PRETRAINED_MODEL=
STEP=0

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
TASK_DIR=${BASE_DIR}task/
RUN_DIR=${BASE_DIR}runs/

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
    esac
done

export LOG_STEM

if [[ $STEP -le 0 ]]; then
    # Training models with the given stem and pretrained models
    ${TASK_DIR}train_all.sh \
        --pretrained "${PRETRAINED_MODEL}" \
        --stem "${LOG_STEM}"
fi

if [[ $STEP -le 1 ]]; then
    # evaluate the original model and get performance metrics and etc.
    ${TASK_DIR}evaluate_all.sh \
        --stem "${LOG_STEM}"
fi

if [[ $STEP -le 2 ]]; then
    # inference and get logits for all tasks
    ${TASK_DIR}predict_all.sh \
        --stem "${LOG_STEM}"

fi

if [[ $STEP -le 3 ]]; then
    # calibrate all the models and evaluate.
    ${TASK_DIR}calibrate_all.sh \
        --stem "${LOG_STEM}"
fi

if [[ $STEP -le 4 ]]; then
    # push result to related sheet.
    ${TASK_DIR}push_result.sh \
        --stem "${LOG_STEM}"
fi
