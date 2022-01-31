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
        type=str, required=True, choices=['udparse', 'wikiann'],
        help='Which encode-predict task are we generaing for.'
    )

    parser.add_argument('--dir', action='store', dest='dir',
        type=str, required=True,
        help='Destination of the generated data_path json files.'
    )

    args = parser.parse_args()

    return args


def main():
    """This generates datasets from pre-configured files.
    """
    args = parse_args()
    os.makedirs(args.dir, exist_ok=True)
    
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

        remapped = {
            'train_data_path': selected['train'],
            'validation_data_path': selected['dev'],
            'test_data_path': selected['test']
        }

        with open(os.path.join(args.dir, 'en-en.json'), 'w', encoding='utf-8') as file_:
            json.dump(remapped, file_)

    if args.task == 'wikiann':
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

        with open(os.path.join(args.dir, 'en-en.json'), 'w', encoding='utf-8') as file_:
            json.dump(configuration, file_)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()