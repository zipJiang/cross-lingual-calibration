"""This script is used to plot metric variation over time
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(
        """Plot the change of metrics over epoch.
        """
    )
    parser.add_argument(
        '--run_dir', action='store', dest='run_dir',
        type=str, required=True, help='Run dir for the data.'
    )
    parser.add_argument(
        '--save_path', action='store', dest='save_path',
        type=str, required=True, help='save_path'
    )
    parser.add_argument(
        '--metric', action='store', dest='metric',
        type=str, required=True, nargs='+',
        help='Which metric to track.'
    )

    return parser.parse_args()


def main():
    """
    """
    args = parse_args()

    metric_dict = {metric: [] for metric in args.metric}

    for filepath in glob(f"{args.run_dir}/*metrics_*"):
        epoch = int(filepath.split('.json')[0].split('_')[-1])
        with open(filepath, 'r', encoding='utf-8') as file_:
            met = json.load(file_)
        for metric in args.metric:
            metric_dict[metric].append((met[metric], epoch))

    # now working on the metric_dict generation
    fig, axes = plt.subplots(1, 1)

    for metric, vals in metric_dict.items():
        vals = sorted(vals, key=lambda x: x[-1], reverse=False)
        axes.plot([v[1] for v in vals], [v[0] for v in vals], label=metric)
    axes.legend()

    fig.savefig(args.save_path)


if __name__ == '__main__':
    main()