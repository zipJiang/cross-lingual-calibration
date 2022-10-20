#!/bin/bash

# set -x
# set -i

BASE_DIR=/brtx/604-nvme2/zpjiang/encode_predict/

# MODEL_LIST="xlm-roberta-large xlm-roberta-base bert-base-multilingual-cased"
MODEL_LIST="xlm-roberta-large"
TASK_LIST="wikiann"
SETTING_LIST="llr"

for model_name in ${MODEL_LIST[@]}; do
    if [[ ${model_name} == "xlm-roberta-large" ]]; then
        prefix="large-xlmr"
    elif [[ ${model_name} == "xlm-roberta-base" ]]; then
        prefix="xlmr"
    elif [[ ${model_name} == "bert-base-multilingual-cased" ]]; then
        prefix="mbert"
    fi

    for task in ${TASK_LIST[@]}; do

        for setting in ${SETTING_LIST[@]}; do
            if [[ ${setting} == "normal" ]]; then
                suffix=
                dconfig=en-en.json
            elif [[ ${setting} == "lr" ]]; then
                suffix="-lr"
                dconfig=sub.json
            else
                suffix="-llr"
                dconfig=subsub.json
            fi
            # sbatch task/train_crf.sh \
            #     --serialization_dir ${prefix}${suffix}-${task} \
            #     --configuration ${BASE_DIR}configs/seq_tagging.jsonnet \
            #     --data_config ${dconfig} \
            #     --task ${task} \
            #     --pretrained ${model_name}

            sbatch task/train_crf.sh \
                --serialization_dir ${prefix}${suffix}-${task}-crf \
                --configuration ${BASE_DIR}configs/seq_tagging.jsonnet \
                --data_config ${dconfig} \
                --task ${task} \
                --pretrained ${model_name} \
                --crf
        done
    done

done