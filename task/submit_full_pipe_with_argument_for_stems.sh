#!/bin/bash

# Running a loop over all possible configurations
# and submit jobs with same set of arguments for
# a bunch of different stems

set -i
set -x

declare -a STEM_LIST=("xlmr" "xlmr-lr" "xlmr-llr" "mbert" "mbert-lr" "mbert-llr")
BASE_DIR="/brtx/604-nvme2/zpjiang/encode_predict/"
TASK_DIR=${BASE_DIR}task/
RUN_DIR=${BASE_DIR}runs/
ARGS=


while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
        --args)
            # Very important that when submitting jobs, should not
            # use double quote as to combine arguments.
            ARGS="$2"
            shift
            shift
            ;;
        --stem_list)
            STEM_LIST=( $2 )
            shift
            shift
            ;;
    esac
done

for stem in "${STEM_LIST[@]}"; do
    sbatch ${TASK_DIR}full_pipe.sh --stem ${stem} ${ARGS}
done