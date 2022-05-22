#!/bin/bash

STEM=

eval "$(conda shell.bash hook)"
conda activate enc-pred
BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
RUN_DIR="${BASE_DIR}runs/"
SCRIPT_DIR="${BASE_DIR}scripts/"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --stem)
            STEM="$2"
            shift
            shift
            ;;
    esac
done

export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"


python3 ${SCRIPT_DIR}push_result.py \
    --runs ${RUN_DIR}${STEM}_pos_tags ${RUN_DIR}${STEM}_deprel ${RUN_DIR}${STEM}_ner \
    --tasks pos_tags deprel ner \
    --precision 5 \
    --sheetname ${STEM}
