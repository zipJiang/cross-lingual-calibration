"""This function generate all training configs (datapath) for
given training tasks. This is properly anonymized to avoid hard
coded path.

 These functional files are designed to only rely on minimum dependency so
 they could run independently.
"""
import argparse
import os
import json
import re


def parse_args():
    parser = argparse.ArgumentParser(
        """Generating all data_path configs.
        """
    )
    parser.add_argument('--task', action='store', dest='task',
        type=str, required=True, choices=['udparse', 'wikiann', 'xnli'],
        help='Which encode-predict task are we generaing for.'
    )

    parser.add_argument('--write_to', action='store', dest='write_to',
        type=str, required=True,
        help='Destination of the generated data_path json files.'
    )
    
    parser.add_argument(
        "--dataset-dir",
        action='store',
        dest='dataset_dir',
        type=str, required=True,
        help="Where to look for the dataset."
    )

    parser.add_argument(
        '--subsample', action='store', dest='subsample',
        choices=["full", "low", "very-low"],
        default="full",
        required=False, help='Whether to use the subsampled config.'
    )

    args = parser.parse_args()

    return args


def main():
    """This generates datasets from pre-configured files.
    """
    args = parse_args()
    
    if args.task == 'udparse':
        # generating all configures according to EIAIT( EMNLP PAPER )
        stem = os.path.join(args.dataset_dir, 'ud-treebanks-v2.9')
        configuration = {
            'train': [
                r'UD_English-.*'
            ],
            'dev': [
                r'UD_English-.*'
            ],
            'test': [
                r'UD_English-PUD'
            ]
        }

        selected = {
            'train': [],
            'dev': [],
            'test': []
        }

        for directory in os.listdir(stem):
            for key in selected:
                for regex in configuration[key]:
                    if re.search(regex, directory):
                        for filename in os.listdir(os.path.join(stem, directory)):
                            if filename.endswith('conllu') and filename.find(key) != -1:
                                selected[key].append(os.path.join(stem, directory, filename))

        if args.subsample == "full":
            remapped = {
                'train_data_path': selected['train'],
                'validation_data_path': selected['dev'],
                'test_data_path': selected['test']
            }

        elif args.subsample == 'low':
            remapped = {
                'train_data_path': os.path.join(args.data_dir, "subsampled_dataset", "subsampled_en_ewt-ud-2.9-train.conllu"),
                'validation_data_path': selected['dev'],
                'test_data_path': selected['test']
            }
        
        else:
            remapped = {
                'train_data_path': os.path.join(args.data_dir, "subsampled_dataset", "subsubsampled_en_ewt-ud-2.9-train.conllu"),
                'validation_data_path': selected['dev'],
                'test_data_path': selected['test']
            }

        with open(args.write_to, 'w', encoding='utf-8') as file_:
            json.dump(remapped, file_, indent=4)

    elif args.task == 'wikiann':

        if args.subsample == "full":
            configuration = {
                'train_data_path': [
                    'en/train'
                ],
                'validation_data_path': [
                    'en/validation'
                ],
                'test_data_path': [
                    'en/test'
                ]
            }
        elif args.subsample == "low":
            configuration = {
                'train_data_path': [
                    'en/train[:1000]'
                ],
                'validation_data_path': [
                    'en/validation'
                ],
                'test_data_path': [
                    'en/test'
                ]
            }
            
        else:
            configuration = {
                'train_data_path': [
                    'en/train[:100]'
                ],
                'validation_data_path': [
                    'en/validation'
                ],
                'test_data_path': [
                    'en/test'
                ]
            }

        with open(args.write_to, 'w', encoding='utf-8') as file_:
            json.dump(configuration, file_)

    elif args.task == 'xnli':
        if args.subsample == "full":
            configuration = {
                'train_data_path': [
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_train.jsonl')
                ],
                'validation_data_path': [
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_dev_mismatched.jsonl'),
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_dev_matched.jsonl')
                ],
                'test_data_path': [
                    os.path.join(args.dataset_dir, 'xnli-test', 'en.jsonl')
                ]
            }
        elif args.subsample == 'low':
            configuration = {
                'train_data_path': [
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_train.jsonl[:1000]')
                ],
                'validation_data_path': [
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_dev_mismatched.jsonl'),
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_dev_matched.jsonl')
                ],
                'test_data_path': [
                    os.path.join(args.dataset_dir, 'xnli-test', 'en.jsonl')
                ]
            }
        else:
            configuration = {
                'train_data_path': [
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_train.jsonl[:100]')
                ],
                'validation_data_path': [
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_dev_mismatched.jsonl'),
                    os.path.join(args.dataset_dir, "multinli_1.0", 'multinli_1.0_dev_matched.jsonl')
                ],
                'test_data_path': [
                    os.path.join(args.dataset_dir, 'xnli-test', 'en.jsonl')
                ]
            }

        with open(args.write_to, 'w', encoding='utf-8') as file_:
            json.dump(configuration, file_, indent=4)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()