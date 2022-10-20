#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

# train_all.sh this script runs the main script train.sh three times to train model for all three tasks (dispatch three tasks)

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"

WIKIANN_CONFIG="${BASE_DIR}configs/wikiann_ner.jsonnet"
UDPARSE_CONFIG="${BASE_DIR}configs/udparse.jsonnet"
XNLI_CONFIG="${BASE_DIR}configs/xnli.jsonnet"
PRETRAINED_MODEL="xlm-roberta-base"
DCONFIG_FILENAME=en-en.json
LEARNING_RATE=0.00001
RECOVER_FLAG=
STEM=

COMMAND=
declare -a TASK_LIST=("wikiann" "xnli" "deprel")


while [[ $# -gt 0 ]]
do
    key="$1"
    echo ":::${key}"
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
            TASK_LIST=( $2 )
            shift
            shift
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        --recover)
            RECOVER_FLAG="--recover"
            shift
            ;;
    esac
done


for task_name in "${TASK_LIST[@]}"; do

    # TODO: set learning rate for different configuration
    if [ ${task_name} == 'udparse' ]; then
        ${COMMAND} ${BASE_DIR}task/train.sh ${RECOVER_FLAG} \
            --serialization_dir "${BASE_DIR}runs/${STEM}_pos_tags" \
            --task "pos_tags" \
            --configuration ${UDPARSE_CONFIG} \
            --data_config "${BASE_DIR}data/udparse_train/${DCONFIG_FILENAME}" \
            --pretrained "${PRETRAINED_MODEL}" \
            --learning_rate "${LEARNING_RATE}"

        # then, the udparse experiments with task deprel
        ${COMMAND} ${BASE_DIR}task/train.sh ${RECOVER_FLAG} \
            --serialization_dir "${BASE_DIR}runs/${STEM}_deprel" \
            --task "deprel" \
            --configuration ${UDPARSE_CONFIG} \
            --data_config "${BASE_DIR}data/udparse_train/${DCONFIG_FILENAME}" \
            --pretrained ${PRETRAINED_MODEL} \
            --learning_rate "${LEARNING_RATE}"

    elif [ ${task_name} == 'wikiann' ]; then
        # then, the ner task
        ${COMMAND} ${BASE_DIR}task/train.sh ${RECOVER_FLAG} \
            --serialization_dir "${BASE_DIR}runs/${STEM}_ner" \
            --task "ner" \
            --configuration ${WIKIANN_CONFIG} \
            --data_config "${BASE_DIR}data/wikiann/data_config/${DCONFIG_FILENAME}" \
            --pretrained ${PRETRAINED_MODEL} \
            --learning_rate "${LEARNING_RATE}"

    elif [ ${task_name} == 'xnli' ]; then
        ${COMMAND} ${BASE_DIR}task/train.sh \
            --serialization_dir "${BASE_DIR}runs/${STEM}_xnli" \
            --task "xnli" \
            --configuration ${XNLI_CONFIG} \
            --data_config "${BASE_DIR}data/XNLI-1.0/data_configs/${DCONFIG_FILENAME}" \
            --pretrained ${PRETRAINED_MODEL} \
            --learning_rate "${LEARNING_RATE}" "${RECOVER_FLAG}"
    fi

done
