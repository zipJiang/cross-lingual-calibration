"""
This script is used to convert ACE data into the
input format of our calibrator.
"""
import json
from typing import Text, Dict, Union, Tuple, List, Optional
import argparse


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Generating calibration input from ace predictions.
        """
    )
    parser.add_argument('--input_path', action='store', dest='input_path',
                        type=str, required=True)
    parser.add_argument('--output_path', action='store', dest='output_path',
                        type=str, required=True)
    parser.add_argument('--depth', action='store', dest='depth',
                        type=int, required=True, nargs='+')

    return parser.parse_args()


def dfs(
    prediction: List,
    depth_selection: List[int],
    sentence: List[Text]) -> List[Dict[Text, Union[List[float], int]]]:
    """This function will push through the predictions to generate final predictions.
    should be selected.
    """
    return_list = []
    if not depth_selection:
        return []
    for span in prediction:
        if min(depth_selection) == 0:
            return_list.append(
                {
                    'parent_label': span['label'],
                    'parent_span': span['span'],
                    'label': [sp['label'] for sp in span['children']],
                    'child_spans': [sp['span'] for sp in span['children']],
                    'sentence': sentence
                }
            )

        return_list.extend(dfs(prediction=span['children'],
                               depth_selection=[d - 1 for d in depth_selection if d > 0],
                               sentence=sentence))

    return return_list


def main():
    """
    """
    args = parse_args()
    depth_selection = args.depth if isinstance(args.depth, list) else [args.depth]

    items = []

    with open(args.input_path, 'r', encoding='utf-8') as file_:
        for line in file_:
            inputs = json.loads(line)['inputs']
            for instance in inputs:
                sentence = instance['sentence']
                selected_items = dfs([instance['spans']], depth_selection, sentence)
            items.extend(selected_items)

    with open(args.output_path, 'w', encoding='utf-8') as file_:
        for it in items:
            file_.write(json.dumps(it) + '\n')


if __name__ == '__main__':
    main()