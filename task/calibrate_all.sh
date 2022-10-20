#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
RUN_DIR=${BASE_DIR}runs/
COMMAND=
STEM=
declare -a TASK_LIST=("wikiann" "xnli" "udparse")
CALIBRATION_STEP=0
# CALIBRATION_MODULE_TYPES=("temperature-scaling" "beta-calibration" "gp-calibration" "histogram-binning")
CALIBRATION_MODULE_TYPES=("temperature-scaling" "gp-calibration")
HINGE='-'


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
        --run_dir)
            RUN_DIR="${BASE_DIR}$2"
            [[ ${RUN_DIR} != "*/" ]] && RUN_DIR=${RUN_DIR}/
            # [[ ${RUN_DIR} == "seqtag-runs/" ]] && HINGE='-'

            shift
            shift
            ;;
    esac
done

# This function will evaluate all items according to the evaluation setting
TASK_DIR=${BASE_DIR}task/
# RUN_DIR=${BASE_DIR}seqtag-runs/

for task_name in "${TASK_LIST[@]}"; do
    declare -a subdir_list=()
    if [ ${task_name} == 'wikiann' ]; then
        subdir_list=("wikiann")
    elif [ ${task_name} == 'udparse' ]; then
        subdir_list=("deprel")
    elif [ ${task_name} == 'xnli' ]; then
        subdir_list=("xnli")
    else
        subdir_list=("${task_name}")
    fi

    for sdir in "${subdir_list[@]}"; do

        for module_name in "${CALIBRATION_MODULE_TYPES[@]}"; do
            ${COMMAND} ${TASK_DIR}calibrate.sh \
                --archive_dir "${RUN_DIR}${STEM}${HINGE}${sdir}" \
                --step ${CALIBRATION_STEP} \
                --module ${module_name}
            if [ ${sdir} == "deprel" ]; then
                ${COMMAND} ${TASK_DIR}calibrate.sh \
                    --archive_dir "${RUN_DIR}${STEM}${HINGE}${sdir}" \
                    --logit_key "selection_logit" \
                    --label_key "selection_label" \
                    --step ${CALIBRATION_STEP} \
                    --module ${module_name}
            fi
        done
    done
done