# Calibrating Zero-shot Cross-lingual (Un-)structured Prediction

This is the experiment code repository for the EMNLP 2022 paper: "Calibrating Zero-shot Cross-lingual (Un-)structured Prediction".

## Introduction

In this paper we investigate model calibration in the setting of zero-shot cross-lingual transfer with large-scale pre-trained language models. The level of model calibration is an important metric for evaluating the trustworthiness of predictive models. There exists an essential need for model calibration when natural language models are deployed in critical tasks. We study different post-training calibration methods in structured and unstructured prediction tasks. We find that models trained with data from the source language become less calibrated when applied to the target language and that calibration errors increase with intrinsic task difficulty and relative sparsity of training data. Moreover, we observe a potential connection between the level of calibration error and an earlier proposed measure of the distance from English to other languages. Finally, our comparison demonstrates that among other methods Temperature Scaling (TS) generalizes well to distant languages, but TS fails to calibrate more complex confidence estimation in structured predictions compared to more expressive alternatives like Gaussian Process Calibration (GPcalib).

## Requirements

The original experiments uses python version `3.8.13`. All pacakge dependencies can be found in `requirements.txt`. to install required pacakges, run
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

There are specific files that might come in handy for reproducing the experiments:

