local dataset_paths = std.parseJson(std.extVar("DATA_PATH"));
local pretrained_model = std.extVar("PRETRAINED_MODEL");
local task = std.extVar('TASK');

# training
local max_length = 512;
local hidden_dim = 300;
local cuda_device = std.parseJson(std.extVar("CUDA_DEVICES"));
local batch_size = std.parseJson(std.extVar("BATCH_SIZE"));
local num_workers = std.parseJson(std.extVar("NUM_WORKERS"));
local num_bins = 100;
local steps = [2, 4, 5, 10, 20];
local learning_rate = std.parseJson(std.extVar("LEARNING_RATE"));

local get_prediction_head(taskname) = {
    type: if taskname == 'deprel' then 'biaffine' else 'linear-span-classification-head',
    with_bias: false,
    label_namespace: if taskname == 'deprel' then 'deprel_labels' else 'labels'
} + if taskname == 'deprel' then {
    activation: {
        type: 'tanh'
    },
    hidden_dim: hidden_dim
} else {};

{
    dataset_reader: {
        type: 'universal-dependency',
        max_length: max_length,
        pretrained_model: pretrained_model,
        task: task
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
        type: 'universal-dependency',
        max_length: max_length,
        pretrained_model: pretrained_model,
        task: task
    },
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

    evaluate_on_test: true,

    model: {
        type: 'enc-predict-lazy',
        word_embedding: {
            token_embedders: {
                'pieces': {
                    type: 'pretrained_transformer',
                    model_name: pretrained_model
                }
            },
        },
        span_extractor: {
            type: 'self_attentive',
        },
        prediction_head: get_prediction_head(task),
        metrics: {
            'performance': {
                type: if task == 'deprel' then 'attachment-logits' else 'dict-categorical'
            },
            'ece': {
                type: if task == 'deprel' then 'ud-calibration' else 'expected-calibration-error',
                [if task =='deprel' then "arc_metric"]: {
                        type: 'expected-calibration-error',
                        num_bins: num_bins,
                        steps: steps,
                    },
                [if task == 'deprel' then "label_metric"]: {
                        type: 'expected-calibration-error',
                        num_bins: num_bins,
                        steps: steps,
                    },
                [if task == 'pos_tags' then 'num_bins']: num_bins,
                [if task == 'pos_tags' then 'steps']: steps,
            },
            # Notice that brier-score calculated here isn't accurate for deprel position (arc_metric)
            'brier-score': {
                type: if task == 'deprel' then 'ud-calibration' else 'brier-score',
                [if task == 'deprel' then "arc_metric"]: {
                    type: 'brier-score'
                },
                [if task == 'deprel' then "label_metric"]: {
                    type: 'brier-score'
                }
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
        validation_metric: '+performance::' + if task == 'deprel' then 'LAS' else 'accuracy',
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