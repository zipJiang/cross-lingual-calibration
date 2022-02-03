local data_path = std.extVar("DATASET_PATH");
local logits_key = std.extVar("LOGIT_KEY");
local labels_key = std.extVar("LABEL_KEY");
local batch_size = std.parseJson(std.extVar("BATCH_SIZE"));
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local test_data_path = std.extVar("TEST_DATA_PATH");
local validation_data_path = std.extVar("VALIDATION_DATA_PATH");
local learning_rate = std.parseJson(std.extVar("LEARNING_RATE"));

# training
local cuda_device = 0;
local num_bins = 100;
local steps = [2, 4, 5, 10, 20];


{
    dataset_reader: {
        type: 'calibration-reader',
        logits_key: logits_key,
        labels_key: labels_key
    },
    data_loader: {
        type: 'multiprocess',
        cuda_device: cuda_device,
        shuffle: true,
        batch_size: batch_size,
    },
    validation_dataset_reader: {
        type: 'calibration-reader',
        logits_key: logits_key,
        labels_key: labels_key
    },
    validation_data_loader: {
        type: 'multiprocess',
        cuda_device: cuda_device,
        shuffle: false,
        batch_size: batch_size,
    },
    evaluate_on_test: true,

    model: {
        type: "calibration-model",
        scaling_module: {
            type: 'temperature-scaling',
        },
        ori_metrics: {
            "ece": {
                type: "expected-calibration-error",
                num_bins: num_bins,
                steps: steps
            },
            "brier": {
                type: "brier-score",
            },
        },
        scaled_metrics: {
            "ece": {
                type: "expected-calibration-error",
                num_bins: num_bins,
                steps: steps
            },
            "brier": {
                type: "brier-score"
            },
        },
    },

    trainer: {
        num_epochs: 256,
        patience: 32,
        cuda_device: cuda_device,
        grad_norm: 5.0,
        learning_rate_scheduler: {
            type: "reduce_on_plateau",
            mode: "max",
            factor: 0.25,
            patience: 8
        },
        validation_metric: "-scaled::ece::ECE",
        optimizer: {
            type: 'huggingface_adamw',
            lr: learning_rate
        },
        chekcpointer: {
            type: 'default',
            keep_most_recent_by_count: 1
        }
    },

    validation_data_path: validation_data_path,
    train_data_path: train_data_path,
    test_data_path: test_data_path,
}