declare -a TASK_LIST=("wikiann" "pos")
declare -a SOURCE_SUFFIX=("" "-lr" "-llr")
declare -a MODEL_STEM=("large-xlmr" "xlmr" "mbert")

for task in "${TASK_LIST[@]}"; do
    for suffix in "${SOURCE_SUFFIX[@]}"; do
        for stem in "${MODEL_STEM[@]}"; do
            sbatch task/predict_all_spans.sh --stem "${stem}${suffix}" --task ${task}
            echo "sbatch task/predict_all_spans.sh --stem ${stem}${suffix} --task ${task}"
        done
    done
done