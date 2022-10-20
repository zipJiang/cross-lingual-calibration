local dataset_paths = std.parseJson(std.extVar("DATA_PATH"));
local pretrained_model = std.extVar("PRETRAINED_MODEL");
local vocab_directory = "/brtx/604-nvme2/zpjiang/encode_predict/data/wikiann/vocabulary";

# training
local max_length = 512;
local hidden_dim = 300;
local cuda_device = std.parseJson(std.extVar("CUDA_DEVICES"));
local batch_size = std.parseJson(std.extVar("BATCH_SIZE"));
local num_workers = std.parseJson(std.extVar("NUM_WORKERS"));
local learning_rate = std.parseJson(std.extVar("LEARNING_RATE"));
local task = std.extVar("SEQTAG_TASK");
local with_crf = std.parseJson(std.extVar('WITH_CRF'));

local get_dataset_reader(task_name, is_prediction) = {
    type: if task_name == 'wikiann' then 'ner-sequence-tagging' else 'pos-tagging-reader',
    pretrained_model: pretrained_model,
    max_length: max_length,
    is_prediction: is_prediction
};

{
    dataset_reader: get_dataset_reader(task, false),

    data_loader: {
        type: 'multiprocess',
        # data_path: dataset_paths['train'],
        num_workers: num_workers,
        cuda_device: cuda_device,
        batch_sampler: {
            type: "max_tokens_sampler",
            max_tokens: max_length * batch_size,
            sorting_keys: ['tokens']
        },
    },

    validation_dataset_reader: get_dataset_reader(task, false),

    validation_data_loader: {
        type: 'multiprocess',
        # data_path: dataset_paths['validation'],
        num_workers: num_workers,
        cuda_device: cuda_device,
        batch_sampler: {
            type: "max_tokens_sampler",
            max_tokens: max_length * batch_size,
            sorting_keys: ['tokens']
        }
    },

    [if task == 'wikiann' then "vocabulary"]: {
        type: "from_files",
        directory: vocab_directory
    },

    evaluate_on_test: true,

    model: {
        type: 'ppcrf',
        with_crf: with_crf,

        word_embedding: {
            token_embedders: {
                'pieces': {
                    type: 'pretrained_transformer',
                    model_name: pretrained_model
                }
            },
        },
        prediction_head: {
            type: 'linear-span-classification-head',
            with_bias: false,
        },
        metrics: {
            'performance': {
                type: 'fbeta',
                average: 'micro',
                [if task == 'wikiann' then "labels"]:  [1, 2, 3, 4, 5, 6]
            },
        },
        initializer: {
            'regexes': [
                [".*_transformers.*linear.*\\.weight", {'mean': 0, 'std': 0.02, 'type': 'normal'}],
                [".*_transformers.*norm.*\\.weight", {'type': 'constant', 'val': 1}],
                [".*_transformers.*\\.bias", {'type': 'zero'}],
            ],
        }
    },

    trainer: {
        num_epochs: 256,
        patience: 8,
        cuda_device: cuda_device,
        grad_norm: 1.0,
        learning_rate_scheduler: {
            type: "reduce_on_plateau",
            mode: "max",
            factor: 0.25,
            patience: 4
        },
        validation_metric: '+performance::fscore',
        optimizer: {
            type: 'huggingface_adamw',
            lr: learning_rate,
            parameter_groups: [
                [['.*transformer_model\\.embeddings.*'], {'requires_grad': false}],
                [['.*_transformer.*'], {'lr': 1.2e-5}],
            ],
        },
        chekcpointer: {
            type: 'default',
            keep_most_recent_by_count: 1
        }
    },
} + dataset_paths