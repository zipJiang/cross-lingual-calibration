#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1

TRAIN_DATA_PATH=
VALIDATION_DATA_PATH=
TEST_DATA_PATH=
LEARNING_RATE=0.1
LABEL_KEY='label'
LOGIT_KEY='logit'
NUM_INDUCING_POINTS=20
NUM_RUNS=10
BATCH_SIZE=4096
CALIBRATION_MODULE_TYPE='temperature-scaling'
ARCHIVE_DIR=
BASE_DIR=$(pwd)/

# CONTROL FLOW
STEP=0

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -l|--learning_rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        # --train_data_path)
        #     TRAIN_DATA_PATH="$2"
        #     shift
        #     shift
        #     ;;
        # --validation_data_path)
        #     VALIDATION_DATA_PATH="$2"
        #     shift
        #     shift
        #     ;;
        # --test_data_path)
        #     TEST_DATA_PATH="$2"
        #     shift
        #     shift
        #     ;;
        --archive_dir)
            ARCHIVE_DIR="$2"
            [[ "${ARCHIVE_DIR}" != */ ]] && ARCHIVE_DIR="${ARCHIVE_DIR}/"
            shift
            shift
            ;;
        --logit_key)
            LOGIT_KEY="$2"
            shift
            shift
            ;;
        --label_key)
            LABEL_KEY="$2"
            shift
            shift
            ;;
        # -s|--serialization_dir)
        #     SERIALIZATION_DIR="$2"
        #     [[ "${SERIALIZATION_DIR}" != */ ]] && SERIALIZATION_DIR="${SERIALIZATION_DIR}/"
        #     shift
        #     shift
        #     ;;
        --base_dir)
            BASE_DIR="$2"
            shift
            shift
            ;;
        --module)
            CALIBRATION_MODULE_TYPE="$2"
            [[ ${CALIBRATION_MODULE_TYPE} == "gp-calibration" ]] && BATCH_SIZE=512
            shift
            shift
            ;;
        --num_inducing_points)
            NUM_INDUCING_POINTS="$2"
            shift
            shift
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift
            shift
            ;;
        --step)
            STEP="$2"
            shift
            shift
            ;;
        --run_id)
            SPECIFIC_RUN_ID="$2"
            shift
            shift
            ;;
        --base_dir)
            BASE_DIR="$2"
            [[ "${BASE_DIR}" != */ ]] && BASE_DIR="${BASE_DIR}/"
            shift
            shift
            ;;
    esac
done

SCRIPT_DIR="${BASE_DIR}scripts/"
export CONGIFURATION_PATH=${BASE_DIR}configs/calibration.jsonnet

export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"
export LEARNING_RATE
export LABEL_KEY
export LOGIT_KEY
export NUM_INDUCING_POINTS
export CALIBRATION_MODULE_TYPE
export BATCH_SIZE

# extract data_file_path from the data_dir
export SERIALIZATION_DIR_BASE="$(dirname ${ARCHIVE_DIR})/$(basename ${ARCHIVE_DIR})=${CALIBRATION_MODULE_TYPE}=${LOGIT_KEY}/"
export DATA_DIR=${ARCHIVE_DIR}calibration/
# export TRAIN_DATA_PATH="${DATA_DIR}calibration-train.jsonl"
export VALIDATION_DATA_PATH="${DATA_DIR}calibration-dev.jsonl"
export TEST_DATA_PATH="${DATA_DIR}en.jsonl"



if [[ ! -z ${SPECIFIC_RUN_ID} ]]; then
    SERIALIZATION_DATA_DIR="${SERIALIZATION_DIR_BASE}data/"
    run_id=${SPECIFIC_RUN_ID}

    SERIALIZATION_DIR="${SERIALIZATION_DIR_BASE}${run_id}/"

    if [[ $CALIBRATION_MODULE_TYPE != "histogram-binning" ]]; then
        if [[ $STEP -le 0 ]]; then
            rm -rf ${SERIALIZATION_DIR}
            python3 ${SCRIPT_DIR}bootstrap.py --src "${DATA_DIR}calibration-train.jsonl" --tgt "${SERIALIZATION_DATA_DIR}calibration-train-${run_id}.jsonl"
            export TRAIN_DATA_PATH="${SERIALIZATION_DATA_DIR}calibration-train-${run_id}.jsonl"

            allennlp train \
                --include-package enc_pred \
                --file-friendly-logging \
                -s "${SERIALIZATION_DIR}" \
                ${CONGIFURATION_PATH}
        fi

        if [[ $STEP -le 1 ]]; then
            mkdir -p ${SERIALIZATION_DIR}eval/
            for filename in $(ls ${DATA_DIR}*.jsonl); do
                filename=$(basename ${filename})
                # change suffix
                if [[ "${filename}" != calibration* ]]; then
                    allennlp evaluate \
                        ${SERIALIZATION_DIR} \
                        "${DATA_DIR}${filename}" \
                        --include-package enc_pred \
                        --file-friendly-logging \
                        --cuda-device 0 \
                        --output-file "${SERIALIZATION_DIR}eval/${filename/jsonl/json}"
                fi
            done
        fi

    else
        items=()
        for filepath in $(ls ${DATA_DIR}*.jsonl); do
            if [[ $(basename ${filepath}) != calibration* ]]; then
                items+=("$filepath")
            fi
        done

        python3 ${SCRIPT_DIR}histogram_binning.py \
            --train_path ${TRAIN_DATA_PATH} \
            --eval_path ${items[@]} \
            --serialization_dir ${SERIALIZATION_DIR} \
            --label_key ${LABEL_KEY} \
            --logit_key ${LOGIT_KEY} \
            --num_bins 100
    fi
else
    # run num_exp times experiments
    rm -rf ${SERIALIZATION_DIR_BASE}
    SERIALIZATION_DATA_DIR="${SERIALIZATION_DIR_BASE}data/"
    mkdir -p ${SERIALIZATION_DATA_DIR}

    for run_id in $(seq ${NUM_RUNS}); do

        SERIALIZATION_DIR="${SERIALIZATION_DIR_BASE}${run_id}/"
        python3 ${SCRIPT_DIR}bootstrap.py --src "${DATA_DIR}calibration-train.jsonl" --tgt "${SERIALIZATION_DATA_DIR}calibration-train-${run_id}.jsonl"
        export TRAIN_DATA_PATH="${SERIALIZATION_DATA_DIR}calibration-train-${run_id}.jsonl"

        if [[ $CALIBRATION_MODULE_TYPE != "histogram-binning" ]]; then
            if [[ $STEP -le 0 ]]; then
                allennlp train \
                    --include-package enc_pred \
                    --file-friendly-logging \
                    -s "${SERIALIZATION_DIR}" \
                    ${CONGIFURATION_PATH}
            fi

            if [[ $STEP -le 1 ]]; then
                mkdir -p ${SERIALIZATION_DIR}eval/
                for filename in $(ls ${DATA_DIR}*.jsonl); do
                    filename=$(basename ${filename})
                    # change suffix
                    if [[ "${filename}" != calibration* ]]; then
                        allennlp evaluate \
                            ${SERIALIZATION_DIR} \
                            "${DATA_DIR}${filename}" \
                            --include-package enc_pred \
                            --file-friendly-logging \
                            --cuda-device 0 \
                            --output-file "${SERIALIZATION_DIR}eval/${filename/jsonl/json}"
                    fi
                done
            fi

        else
            items=()
            for filepath in $(ls ${DATA_DIR}*.jsonl); do
                if [[ $(basename ${filepath}) != calibration* ]]; then
                    items+=("$filepath")
                fi
            done

            python3 ${SCRIPT_DIR}histogram_binning.py \
                --train_path ${TRAIN_DATA_PATH} \
                --eval_path ${items[@]} \
                --serialization_dir ${SERIALIZATION_DIR} \
                --label_key ${LABEL_KEY} \
                --logit_key ${LOGIT_KEY} \
                --num_bins 100
        fi
    done
fi