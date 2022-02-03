# !/bin/bash
# script running prediction logits over all the lang and task pairs.

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
COMMAND=


while [[ $# -gt 0 ]];
do
    key="$1"
    case $key in
        --distributed)
            COMMAND=sbatch
            shift
            ;;
    esac
done

# This function will evaluate all items according to the evaluation setting
# To specify a language called calibration will predict over calibration tuning set.
declare -a LANG_LIST=("ar" "de" "en" "es" "fr" "hi" "ru" "zh" "calibration-train" "calibration-dev")
declare -a TASK_LIST=("wikiann" "udparse")

TASK_DIR=${BASE_DIR}task/
RUN_DIR=${BASE_DIR}runs/


for lang in "${LANG_LIST[@]}"; do
    for task in "${TASK_LIST[@]}"; do
    declare -a model_dirs=()
        if [ ${task} = "udparse" ]; then
            model_dirs=("${RUN_DIR}pos_tags_en_en" "${RUN_DIR}deprel_en_en")
        elif [ ${task} == 'wikiann' ]; then
            model_dirs=("${RUN_DIR}ner_en_en")
        fi

        for sdir in "${model_dirs[@]}"; do
            ${COMMAND} ${TASK_DIR}predict_logits.sh \
                --lang ${lang} \
                --task ${task} \
                --serialization_dir ${sdir}
        done
    done
done