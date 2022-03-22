local dataset_paths = std.parseJson(std.extVar("DATA_PATH"));
local pretrained_model = std.extVar("PRETRAINED_MODEL");

# training
local max_length = 128;
local hidden_dim = 300;
local cuda_device = std.parseJson(std.extVar("CUDA_DEVICES"));
local batch_size = std.parseJson(std.extVar("BATCH_SIZE"));
local num_workers = std.parseJson(std.extVar("NUM_WORKERS"));
local num_bins = 100;
local steps = [2, 4, 5, 10, 20];
local learning_rate = std.parseJson(std.extVar("LEARNING_RATE"));
local epochs = std.parseJson(std.extVar("EPOCHS"));
local patience = std.parseJson(std.extVar("PATIENCE"));

{
    dataset_reader: {
        type: 'xnli',
        max_length: max_length,
        pretrained_model: pretrained_model
    },
    data_loader: {
        type: 'multiprocess',
        # data_path: dataset_paths['train'],
        num_workers: num_workers,
        cuda_device: cuda_device,
        batch_sampler: {
            type: "max_tokens_sampler",
            max_tokens: max_length * batch_size,
            sorting_keys: ['tokens']
        }
    },
    validation_dataset_reader: {
        type: 'xnli',
        max_length: max_length,
        pretrained_model: pretrained_model
    },
    validation_data_loader: {
        type: 'multiprocess',
        # data_path: dataset_paths['validation'],
        num_workers: num_workers,
        cuda_device: cuda_device,
        batch_sampler: {
            type: "max_tokens_sampler",
            max_tokens: max_length * batch_size * 2,
            sorting_keys: ['tokens']
        }
    },

    evaluate_on_test: true,

    model: {
        type: 'enc-sentpred-lazy',
        word_embedding: {
            token_embedders: {
                'pieces': {
                    type: 'pretrained_transformer',
                    model_name: pretrained_model
                }
            },
        },
        seq2vec_encoder: {
            type: 'bert_pooler',
            pretrained_model: pretrained_model,
        },
        prediction_head: {
            type: 'linear-classification-head',
            with_bias: false,
            label_namespace: 'labels'
        },
        metrics: {
            'performance': {
                type: 'dict-categorical',
            },
            'ece': {
                type: 'expected-calibration-error',
                num_bins: num_bins,
                steps: steps
            },
            'brier-score': {
                type: 'brier-score'
            }
        },
        # Only useful when there is transformer models.
        initializer: {
            'regexes': [
                [".*_transformers.*linear.*\\.weight", {'mean': 0, 'std': 0.02, 'type': 'normal'}],
                [".*_transformers.*norm.*\\.weight", {'type': 'constant', 'val': 1}],
                [".*_transformers.*\\.bias", {'type': 'zero'}],
            ],
        }
    },
    trainer: {
        num_epochs: epochs,
        patience: patience,
        cuda_device: cuda_device,
        grad_norm: 1.0,
        learning_rate_scheduler: {
            type: "reduce_on_plateau",
            mode: "max",
            factor: 0.5,
            patience: 16
        },
        validation_metric: '+performance::accuracy',
        optimizer: {
            type: 'huggingface_adamw',
            lr: learning_rate,
            parameter_groups: [
                [['.*prediction_head.*'], {'lr': 1e-4}],
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
