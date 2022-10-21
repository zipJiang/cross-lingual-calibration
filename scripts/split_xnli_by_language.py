"""This script will split the xnli data to
small dumps by languages.
"""
import argparse
import os
import json


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Splitting XNLI data to languages.
        """
    )
    parser.add_argument(
        '--file_path', action='store', dest='file_path',
        type=str, required=True, help='Where to load the files.'
    )
    parser.add_argument(
        '--dump_dir', action='store', dest='dump_dir',
        type=str, required=True, help='Where to dump the generated files.'
    )
    
    return parser.parse_args()


def main():
    """
    """
    args = parse_args()

    lang_file_lists = {}

    with open(args.file_path, 'r', encoding='utf-8') as file_:
        for line in file_:
            item = json.loads(line)
            if item['language'] not in lang_file_lists:
                lang_file_lists[item['language']] = []
            lang_file_lists[item['language']].append(item)

    os.makedirs(args.dump_dir, exist_ok=True)

    for lang, item_list in lang_file_lists.items():
        with open(os.path.join(args.dump_dir, f"{lang}.jsonl"), 'w', encoding='utf-8') as file_:
            for item in item_list:
                file_.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    main()