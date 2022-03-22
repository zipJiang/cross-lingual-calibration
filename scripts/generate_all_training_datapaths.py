"""This function generate all training configs (datapath) for
given training tasks.
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
        '--subsample', action='store_true', dest='subsample',
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
        stem = '/brtx/604-nvme2/zpjiang/encode_predict/data/universal_dependency/ud-treebanks-v2.9'
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

        if not args.subsample:
            remapped = {
                'train_data_path': selected['train'],
                'validation_data_path': selected['dev'],
                'test_data_path': selected['test']
            }

        else:
            remapped = {
                'train_data_path': "/brtx/604-nvme2/zpjiang/encode_predict/data/udparse_train/subsampled_datasets/subsampled_en_ewt-ud-2.9-train.conllu",
                'validation_data_path': selected['dev'],
                'test_data_path': selected['test']
            }

        with open(args.write_to, 'w', encoding='utf-8') as file_:
            json.dump(remapped, file_)

    elif args.task == 'wikiann':

        if not args.subsample:
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
        else:
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

        with open(args.write_to, 'w', encoding='utf-8') as file_:
            json.dump(configuration, file_)

    elif args.task == 'xnli':
        if not args.subsample:
            configuration = {
                'train_data_path': [
                    '/brtx/604-nvme2/zpjiang/encode_predict/data/multinli_1.0/multinli_1.0_train.jsonl'
                ],
                'validation_data_path': [
                    '/brtx/604-nvme2/zpjiang/encode_predict/data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl',
                    '/brtx/604-nvme2/zpjiang/encode_predict/data/multinli_1.0/multinli_1.0_dev_matched.jsonl'
                ],
                'test_data_path': [
                    '/brtx/604-nvme2/zpjiang/encode_predict/data/xnli-test/en.jsonl'
                ]
            }
        else:
            configuration = {
                'train_data_path': [
                    '/brtx/604-nvme2/zpjiang/encode_predict/data/multinli_1.0/multinli_1.0_train.jsonl[:1000]'
                ],
                'validation_data_path': [
                    '/brtx/604-nvme2/zpjiang/encode_predict/data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl',
                    '/brtx/604-nvme2/zpjiang/encode_predict/data/multinli_1.0/multinli_1.0_dev_matched.jsonl'
                ],
                'test_data_path': [
                    '/brtx/604-nvme2/zpjiang/encode_predict/data/xnli-test/en.jsonl'
                ]
            }

        with open(args.write_to, 'w', encoding='utf-8') as file_:
            json.dump(configuration, file_)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()