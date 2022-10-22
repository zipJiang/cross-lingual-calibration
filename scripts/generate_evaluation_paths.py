"""This script will generate paths to all evaluation scripts
that serves the multi-lingual evaluation.
"""
import argparse
import os
import json
from typing import Text, List, Union, Optional, Dict, Callable


def parse_args():
    parser = argparse.ArgumentParser(
        """Prepare evaluation path for multi-lingual settings.
        """
    )

    # Here task actually corresponds to dataset
    parser.add_argument(
        '--task', action='store', dest='task',
        type=str, required=True, help='Which task to represent.',
        choices=['udparse', 'wikiann', 'xnli'])

    parser.add_argument(
        '--lang', action='store', dest='lang',
        type=str, required=True, help='Which language dataset to use.'
    )
    parser.add_argument(
        '--return_dict', action='store_true', dest='return_dict',
        help='Whether to return a jsonl dict.'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == 'udparse':
        # For open sourced version we will use a slightly different dir structure
        config = {
            'ar': 'data/ud-treebanks-v2.9/UD_Arabic-PUD/ar_pud-ud-test.conllu',
            'de': 'data/ud-treebanks-v2.9/UD_German-PUD/de_pud-ud-test.conllu',
            'en': 'data/ud-treebanks-v2.9/UD_English-PUD/en_pud-ud-test.conllu',
            'es': 'data/ud-treebanks-v2.9/UD_Spanish-PUD/es_pud-ud-test.conllu',
            'fr': 'data/ud-treebanks-v2.9/UD_French-PUD/fr_pud-ud-test.conllu',
            'hi': 'data/ud-treebanks-v2.9/UD_Hindi-PUD/hi_pud-ud-test.conllu',
            'ru': 'data/ud-treebanks-v2.9/UD_Russian-PUD/ru_pud-ud-test.conllu',
            'zh': 'data/ud-treebanks-v2.9/UD_Chinese-PUD/zh_pud-ud-test.conllu',
            'calibration-train': [
                'data/ud-treebanks-v2.9/UD_English-Atis/en_atis-ud-test.conllu',
                'data/ud-treebanks-v2.9/UD_English-EWT/en_ewt-ud-test.conllu',
                'data/ud-treebanks-v2.9/UD_English-GUM/en_gum-ud-test.conllu',
                'data/ud-treebanks-v2.9/UD_English-LinES/en_lines-ud-test.conllu',
                'data/ud-treebanks-v2.9/UD_English-ParTUT/en_partut-ud-test.conllu',
                'data/ud-treebanks-v2.9/UD_English-Pronouns/en_pronouns-ud-test.conllu'
            ],
            'calibration-dev': [
                'data/ud-treebanks-v2.9/UD_English-Atis/en_atis-ud-dev.conllu',
                'data/ud-treebanks-v2.9/UD_English-EWT/en_ewt-ud-dev.conllu',
                'data/ud-treebanks-v2.9/UD_English-GUM/en_gum-ud-dev.conllu',
                'data/ud-treebanks-v2.9/UD_English-LinES/en_lines-ud-dev.conllu',
                'data/ud-treebanks-v2.9/UD_English-ParTUT/en_partut-ud-dev.conllu',
            ]
        }

    elif args.task == 'wikiann':
        config = {
            'ar': 'ar/test',
            'de': 'de/test',
            'en': 'en/test',
            'es': 'es/test',
            'fr': 'fr/test',
            'hi': 'hi/test',
            'ru': 'ru/test',
            'zh': 'zh/test',
            'calibration-train': 'en/validation[:3000]',
            'calibration-dev': 'en/validation[3000:]'
        }
    elif args.task == 'xnli':
        config = {
            'ar': 'data/xnli-test/ar.jsonl',
            'de': 'data/xnli-test/de.jsonl',
            'en': 'data/xnli-test/en.jsonl',
            'es': 'data/xnli-test/es.jsonl',
            'fr': 'data/xnli-test/fr.jsonl',
            'hi': 'data/xnli-test/hi.jsonl',
            'ru': 'data/xnli-test/ru.jsonl',
            'zh': 'data/xnli-test/zh.jsonl',
            'calibration-train': [
                'data/multinli_1.0/multinli_1.0_dev_matched.jsonl',
                'data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
            ],
            'calibration-dev': [
                'data/xnli-dev/en.jsonl'
            ]
        }
    else:
        raise NotImplementedError

    assert args.lang in config, f'requested language: {args.lang} not in {config.keys()}!'

    if args.return_dict:
        # We don't have to check conditions because the original data_reader allwos list input.
        print(json.dumps({'file_path': [config[args.lang]]}))
    else:
        return_val = config[args.lang] if isinstance(config[args.lang], str) else ' '.join(config[args.lang])
        print(config[args.lang])


if __name__ == '__main__':
    main()