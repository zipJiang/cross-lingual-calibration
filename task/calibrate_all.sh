# !/bin/bash

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
COMMAND=
STEM=


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
    esac
done

# This function will evaluate all items according to the evaluation setting
declare -a MODEL_LIST=("deprel" "pos_tags" "ner")

TASK_DIR=${BASE_DIR}task/
RUN_DIR=${BASE_DIR}runs/


for sdir in "${MODEL_LIST[@]}"; do

    ${COMMAND} ${TASK_DIR}calibrate.sh \
        --archive_dir "${RUN_DIR}${STEM}_${sdir}" \
        --step 0
    if [ ${sdir} = deprel ]; then
        ${COMMAND} ${TASK_DIR}calibrate.sh \
            --archive_dir "${RUN_DIR}${STEM}_${sdir}" \
            --logit_key "selection_logit" \
            --label_key "selection_label" \
            --step 0
    fi
done
