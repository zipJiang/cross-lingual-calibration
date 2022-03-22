"""This scripts take a calibration
metric script and generate final
metrics that could be applied
to the table
"""
import json
import argparse


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Reading experiment result.
        """
    )
    parser.add_argument(
        '--metric_path', action='store', dest='metric_path',
        type=str, required=True, help='Read metric path.'
    )

    parser.add_argument(
        '--stem', action='store', dest='stem',
        type=str, required=False, default='test'
    )

    return parser.parse_args()


def main():
    """
    """
    args = parse_args()

    with open(args.metric_path, 'r', encoding='utf-8') as file_:
        metrics = json.load(file_)

        if f'{args.stem}_ori::ece::ECE' in metrics:
            print(
                {
                    'ori-ECE': metrics[f'{args.stem}_ori::ece::ECE'],
                    'delta': metrics[f'{args.stem}_scaled::ece::ECE'] - metrics[f'{args.stem}_ori::ece::ECE']
                }
            )
        else:
            print('Only one metric detected (no test val).')
            print(
                {
                    'ori-ECE': metrics['ori::ece::ECE'],
                    'delta': metrics['scaled::ece::ECE'] - metrics['ori::ece::ECE']
                }
            )


if __name__ == '__main__':
    main()