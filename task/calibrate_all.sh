#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
COMMAND=
STEM=
declare -a TASK_LIST=("wikiann" "xnli" "udparse")
CALIBRATION_STEP=0


while [[ $# -gt 0 ]];
do
    key="$1"
    case $key in
        --distributed)
            COMMAND=sbatch
            shift
            ;;
        --stem)
            STEM="$2"
            shift
            shift
            ;;
        --task)
            TASK_LIST=( $2 )
            shift
            shift
            ;;
        --step)
            CALIBRATION_STEP="$2"
            shift
            shift
            ;;
    esac
done

# This function will evaluate all items according to the evaluation setting
TASK_DIR=${BASE_DIR}task/
RUN_DIR=${BASE_DIR}runs/

for task_name in "${TASK_LIST[@]}"; do
    declare -a subdir_list=()
    if [ ${task_name} == 'wikiann' ]; then
        subdir_list=("ner")
    elif [ ${task_name} == 'udparse' ]; then
        subdir_list=("deprel" "pos_tags")
    elif [ ${task_name} == 'xnli' ]; then
        subdir_list=("xnli")
    fi

    for sdir in "${subdir_list[@]}"; do

        ${COMMAND} ${TASK_DIR}calibrate.sh \
            --archive_dir "${RUN_DIR}${STEM}_${sdir}" \
            --step ${CALIBRATION_STEP}
        if [ ${sdir} == "deprel" ]; then
            ${COMMAND} ${TASK_DIR}calibrate.sh \
                --archive_dir "${RUN_DIR}${STEM}_${sdir}" \
                --logit_key "selection_logit" \
                --label_key "selection_label" \
                --step ${CALIBRATION_STEP}
        fi
    done
done
