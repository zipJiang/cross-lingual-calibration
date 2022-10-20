#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

# script running prediction logits over all the lang and task pairs.
set -i
set -x

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
COMMAND=
STEM=
declare -a TASK_LIST=("wikiann" "pos")


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
    esac
done

# This function will evaluate all items according to the evaluation setting
# To specify a language called calibration will predict over calibration tuning set.
declare -a LANG_LIST=("ar" "de" "en" "es" "fr" "hi" "ru" "zh" "calibration-train" "calibration-dev")

SCRIPT_DIR=${BASE_DIR}scripts/
RUN_DIR=${BASE_DIR}seqtag-runs/

for lang in "${LANG_LIST[@]}"; do
    for task in "${TASK_LIST[@]}"; do
        declare -a model_dirs=()
        model_dirs=("${RUN_DIR}${STEM}-${task}/" "${RUN_DIR}${STEM}-${task}-crf/")

        if [[ ${task} == 'pos' ]]; then
            task='udparse'
        fi

        input_file_path=$(mktemp)

        python3 ${SCRIPT_DIR}generate_evaluation_paths.py --lang ${lang} --task ${task} --return_dict > ${input_file_path}

        for sdir in "${model_dirs[@]}"; do
            [ ! -d "${sdir}calibration" ] && mkdir -p ${sdir}calibration
            conda run --no-capture-output -n enc-pred python3 ${SCRIPT_DIR}predict_span_probs.py \
                --data_config_path ${input_file_path} \
                --device 0 \
                --output_path ${sdir}calibration/${lang}.jsonl \
                --archive_path ${sdir}
        done
    done
done
