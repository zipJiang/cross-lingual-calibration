# Calibrating Zero-shot Cross-lingual (Un-)structured Prediction

This is the experiment code repository for the EMNLP 2022 paper: "Calibrating Zero-shot Cross-lingual (Un-)structured Prediction".

## Introduction

In this paper we investigate model calibration in the setting of zero-shot cross-lingual transfer with large-scale pre-trained language models. The level of model calibration is an important metric for evaluating the trustworthiness of predictive models. There exists an essential need for model calibration when natural language models are deployed in critical tasks. We study different post-training calibration methods in structured and unstructured prediction tasks. We find that models trained with data from the source language become less calibrated when applied to the target language and that calibration errors increase with intrinsic task difficulty and relative sparsity of training data. Moreover, we observe a potential connection between the level of calibration error and an earlier proposed measure of the distance from English to other languages. Finally, our comparison demonstrates that among other methods Temperature Scaling (TS) generalizes well to distant languages, but TS fails to calibrate more complex confidence estimation in structured predictions compared to more expressive alternatives like Gaussian Process Calibration (GPcalib).

## Requirements

The original experiments uses python version `3.8.13`. The running shellscript in this repo usually assumes a runnable command `python3`. All pacakge dependencies can be found in `requirements.txt`. to install required pacakges, run
```shellscript
$ pip install -r requirements.txt
```
Notice that due to an issue with `allennlp` you probably need to install `jsonnet` with conda.
To run the code base locally you should add this working directory to `$PYTHONPATH`:
```shellscript
$ cd <path-to-this-repository-directory>
$ export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Preparing Dataset

To prepare the dataset for the multilingual experiments, run the following script:

```shellscript
$ task/prepare_dataset.sh
```

this will generate a dataset directory and donwloading and extracting all relevant dataset.

## File Structures

```shellscript
.
├── configs  # Training configs for different tasks and calibration
├── enc_pred  # Main library for modeling
├── readme.md
├── requirements.txt
├── scripts  # Runnable .py scripts for singular tasks
└── task  # experiment task bashscripts
```

## Running Experiments

### Generating Data Configs
To run experiments for specific dataset with corresponding model configuration. First we need to generate dataset configuration (`full-data`, `low-data`, `very-low-data` as in the paper). There is a particular script in the `scripts/` directory that can help generating data configurations that aligns with the paper.

```shellscript
python3 scripts/generate_all_training_datapaths.py
usage: Generating all data_path configs.
         [-h] --task
                                                  {udparse,wikiann,xnli}
                                                  --write_to WRITE_TO
                                                  --dataset-dir DATASET_DIR
                                                  [--subsample {full,low,very-low}]

optional arguments:
  -h, --help            show this help message and exit
  --task {udparse,wikiann,xnli}
                        Which encode-predict task are we generaing for.
  --write_to WRITE_TO   Destination of the generated data_path json files.
  --dataset-dir DATASET_DIR
                        Where to look for the dataset.
  --subsample {full,low,very-low}
                        Whether to use the subsampled config.

```

### Training a Model with *allennlp* .jsonnet Configs

Training configuration files lies in the configuration directory `configs/`. The name of these configuration files should be self-explanatory. Specifically, `seq_tagging.jsonnet` specify the training configs for structure prediction tasks as in section 4.3, `calibration.jsonnet` specify calibration training configurations, and other configuration files are for section 4.1 and 4.2. To specify hyperparameters for these training hyperparameters, one needs to start training with a script in the `task/` directory:
```shellscript
$ task/train.sh --serialization_dir [SERIALIZATION_DIR] \
    --pretrained [PRETRAINED_MODEL_NAME] \ 
    --configuration [JSONNET TRAINING CONFIG FILES] \
    --data_config [DATA_CONFIG_FILES] \
    --task {pos_tags, deprel, ner, xnli}
```

Where data_config files are the files generated in the previous step. Notice that the configuration file, data\_config file and task should be a compatible triplet.

### Evaluate Model to Generate Raw Logits

After the training finishes and is saved to the serialization directory, We can evaluate the model on the dev set to get training statistics for the original task. This is done using the evaluation shellscript `task/evaluate.sh`:
```shellscript
$ task/evaluate.sh --serialization_dir [SERIALIZATION_DIR] \
    --task {pos_tags, deprel, ner, xnli} \
    --lang {ar, de, en, es, fr, hi, ru, zh, calibration-train, calibration-dev}
```

Notice that the evaluation result contains evaluated ECE for original logits, but to evaluate calibration we still need to get to align predictions with relevant labels this is done in the next step.

### Predict Logits on Language Data

To train a calibrator as described in the next step, a minimum requirement is that you should have logits predictions on `en`, `calibration-train`, and `calibration-dev`. Each evaluation run will generate a file `eval/{lang}.json` in the serialization directory.

```shellscript
$ task/predict_logit.sh \
    --task {pos_tags, deprel, ner, xnli} \
    --serialization_dir [SERIALIZATION_DIR] \
    --lang {ar, de, en, es, fr, hi, ru, zh, calibration-train, calibration-dev}
```

This will generate a subdir `calibration` in the `serialization_dir` that can be read by the calibrator.

### Calibrate a model with the `calibration.jsonnet` config

Suppose we now have all the language files we are interested in the `calibration` subdirectory, one can train a calibrator using the evaluation result with the `task/calibrate.sh`

```shellscript
$ task/calibrate.sh --archive_dir [MODEL_SERIALIZATION_DIR] \
    --logit_key {logit, selection_logit} \
    --label_key {label, selection_label} \
    --module {temperature-scaling, gp-calibration, beta-calibration, histogram-binning} \
    --num_runs [NUM_RUNS]
```

This should sequentially run `NUM_RUNS` times of calibration with bootstrapping data split. For model stored in `MODEL_SERIALIZATION_DIR`, the calibrators will be stored sequentially in `${MODEL_SERIALIZATION_DIR}=${LOGIT_KEY}=${LABEL_KEY}`. Notice that we will also get calibration evaluation result on all language data prepared at the previous step at the same time.
