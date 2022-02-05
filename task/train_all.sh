# !/bin/bash
# train_all.sh this script runs the main script train.sh three times to train model for all three tasks (dispatch three tasks)
set -i
set -x

BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"

WIKIANN_CONFIG="${BASE_DIR}configs/wikiann_ner.jsonnet"
UDPARSE_CONFIG="${BASE_DIR}configs/udparse.jsonnet"
PRETRAINED_MODEL="xlm-roberta-base"
STEM=

COMMAND=


while [[ $# -gt 0 ]]
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
        -p|--pretrained)
            PRETRAINED_MODEL="$2"
            shift
            shift
            ;;
    esac
done


# first, the udparse experiments with task pos_tags
${COMMAND} ${BASE_DIR}task/train.sh \
    --serialization_dir "${BASE_DIR}runs/${STEM}_pos_tags" \
    --task "pos_tags" \
    --configuration ${UDPARSE_CONFIG} \
    --data_config "${BASE_DIR}data/udparse_train/en-en.json" \
    --pretrained "${PRETRAINED_MODEL}"

# then, the udparse experiments with task deprel
${COMMAND} ${BASE_DIR}task/train.sh \
    --serialization_dir "${BASE_DIR}runs/${STEM}_deprel" \
    --task "deprel" \
    --configuration ${UDPARSE_CONFIG} \
    --data_config "${BASE_DIR}data/udparse_train/en-en.json" \
    --pretrained ${PRETRAINED_MODEL}

# then, the ner task
${COMMAND} ${BASE_DIR}task/train.sh \
    --serialization_dir "${BASE_DIR}runs/${STEM}_ner" \
    --task "ner" \
    --configuration ${WIKIANN_CONFIG} \
    --data_config "${BASE_DIR}data/wikiann/data_config/en-en.json" \
    --pretrained ${PRETRAINED_MODEL}