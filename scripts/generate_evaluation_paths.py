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
    parser.add_argument(
        '--task', action='store', dest='task',
        type=str, required=True, help='Which task to represent.',
        choices=['udparse', 'wikiann'])

    parser.add_argument(
        '--lang', action='store', dest='lang',
        type=str, required=True, help='Which language dataset to use.'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    stem = '/brtx/604-nvme2/zpjiang/encode_predict/'

    if args.task == 'udparse':
        config = {
            'ar': 'data/universal_dependency/ud-treebanks-v2.9/UD_Arabic-PUD/ar_pud-ud-test.conllu',
            'de': 'data/universal_dependency/ud-treebanks-v2.9/UD_German-PUD/de_pud-ud-test.conllu',
            'en': 'data/universal_dependency/ud-treebanks-v2.9/UD_English-PUD/en_pud-ud-test.conllu',
            'es': 'data/universal_dependency/ud-treebanks-v2.9/UD_Spanish-PUD/es_pud-ud-test.conllu',
            'fr': 'data/universal_dependency/ud-treebanks-v2.9/UD_French-PUD/fr_pud-ud-test.conllu',
            'hi': 'data/universal_dependency/ud-treebanks-v2.9/UD_Hindi-PUD/hi_pud-ud-test.conllu',
            'ru': 'data/universal_dependency/ud-treebanks-v2.9/UD_Russian-PUD/ru_pud-ud-test.conllu',
            'zh': 'data/universal_dependency/ud-treebanks-v2.9/UD_Chinese-PUD/zh_pud-ud-test.conllu'
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
            'zh': 'zh/test'
        }
    else:
        raise NotImplementedError

    assert args.lang in config, f'requested language: {args.lang} not in {config.keys()}!'
    print(config[args.lang])


if __name__ == '__main__':
    main()