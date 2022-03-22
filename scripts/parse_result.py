"""This function parse result into a json file that
it is easy to generate final result table in latex.
"""
import os
import json
from enc_pred.utils.spreadsheet import TYPE_TO_FEATURES
import argparse


RUN_DIR = '/brtx/604-nvme2/zpjiang/encode_predict/runs'
CNAME_TO_TASK = {
    'deprel': 'UDP',
    'pos_tags': 'POS',
    'ner': 'NER',
    'xnli': 'XNLI'
}


def parse_args():
    parser = argparse.ArgumentParser(
        """Parse result into easily processable json.
        """
    )
    parser.add_argument(
        '--stem', action='store', dest='stem',
        type=str, required=True
    )
    parser.add_argument(
        '--write_to', action='store', dest='write_to',
        type=str, required=True
    )

    return parser.parse_args()


def main():
    """
    """
    args = parse_args()

    result_dict = {}

    for cname, configs in TYPE_TO_FEATURES.items():
        taskname = CNAME_TO_TASK[cname]
        for config in configs:
            suffix, content = config['suffix'], config['content']
            eval_dir = os.path.join(RUN_DIR, f'{args.stem}_{cname}' + suffix, 'eval')
            for eval_name in os.listdir(eval_dir):
                with open(os.path.join(eval_dir,eval_name), 'r', encoding='utf-8') as file_:
                    lang = eval_name.split('.')[0]
                    if lang not in result_dict:
                        result_dict[lang] = {}
                    if taskname not in result_dict[lang]:
                        result_dict[lang][taskname] = {}
                    eval_dict = json.load(file_)

                    for entry in content:
                        label = entry['label']
                        keys = entry['keys']
                        result_dict[lang][taskname][label] = {}

                        for keyname, rk in zip(keys, ['before', 'after']):
                            result_dict[lang][taskname][label][rk] = eval_dict[keyname]

    with open(args.write_to, 'w', encoding='utf-8') as file_:
        json.dump(result_dict, file_, indent=4)


if __name__ == '__main__':
    main()