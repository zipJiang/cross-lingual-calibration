
"""This is the function that calls the allennlp predictor
with model from archive, and generate original logits prediction
from the model.
"""
from allennlp.predictors.predictor import Predictor

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        """Predicting logits for instances with trained models.
        """
    )
    parser.add_argument(
        '--archive_path', action='store', dest='archive_path',
        type=str, required=True, help='Where to load the archive'
    )
    parser.add_argument(
        '--input_path', action='store', dest='input_path',
        type=str, required=True, help='Input path to read config.'
    )
    parser.add_argument(
        '--output_path', action='store', dest='output_path',
        type=str, required=True, help='Output path to write result to.'
    )
    parser.add_argument(
        '--cuda_devices', action='store', dest='cuda_devices',
        type=int, required=True, help='Which device to run prediction on.'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    predictor = Predictor.from_path(
        archive_path=args.archive_path,
        predictor_name='calibration-predictor',
        cuda_device=args.cuda_devices
    )

    with open(args.input_path, 'r', encoding='utf-8') as file_:
        inputs = json.load(file_)

    with open(args.output_path, 'w', encoding='utf-8') as file_:

        for file_path in inputs['file_path']:
            return_list = predictor.predict(
                file_path=file_path
            )

            for item in return_list:
                print(json.dumps(item), file=file_)


if __name__ == '__main__':
    main()