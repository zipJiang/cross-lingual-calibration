"""This scripts plot result that compares
different calibration methods.
"""
import argparse
import glob
from http.client import METHOD_NOT_ALLOWED
import numpy as np
import os
import json
from enc_pred.utils.spreadsheet import CalibrationExperiment, PredictionExperiment, \
    ExperimentSet, ExperimentDataSizeGroup, ExperimentModelGroup, ExperimentTaskGroup
import matplotlib.pyplot as plt


RUN_DIR = '/brtx/604-nvme2/zpjiang/encode_predict/runs/'
LANG_SEQ = ['en', 'de', 'fr', 'es', 'ru', 'hi', 'ar', 'zh']
COLORS = [
    'tomato',
    'seagreen',
    'skyblue',
    'orange'
]

METHOD_LEGEND_MAP = {
    'temperature-scaling': 'TS',
    'gp-calibration': 'GP',
    'beta-calibration': 'Beta',
    'histogram-binning': 'HIST'
}

WIDTH = 3.
STEP_SIZE = 6.


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Try different calibration methods
        that can be used for post-training calibration.
        """
    )
    parser.add_argument(
        '--method', action='store', dest='method',
        type=str, required=False, help='Which methods to compare.',
        nargs='+',
        default=['temperature-scaling', 'beta-calibration', 'histogram-binning', 'gp-calibration']
    )
    parser.add_argument(
        '--stem', action='store', dest='stem', type=str, required=True,
        help='Which model to load.'
    )
    parser.add_argument(
        '--source', action='store', dest='source',
        type=str, required=False, default='logit', help='Which source to load.'
    )
    parser.add_argument(
        '--task', action='store', dest='task',
        type=str, required=True, help='Which task to consider'
    )
    
    parser.add_argument(
        '--output_path', action='store', dest='output_path',
        type=str, required=True, help='Where write the output.'
    )
    
    return parser.parse_args()

    
def main():
    """
    """
    args = parse_args()
    
    prediction_experiment = PredictionExperiment.from_dir(
        dirname=os.path.join(RUN_DIR, f"{args.stem}_{args.task}")
    )

    calibration_experiments = {
        key: CalibrationExperiment.from_dir(
            dirname=os.path.join(RUN_DIR, f"{args.stem}_{args.task}={key}={args.source}"),
        ) for key in args.method
    }

    experiment_set = ExperimentSet(
        prediction_experiment=prediction_experiment,
        calibration_experiments=calibration_experiments
    )
    
    # plot experiment_set result.
    table_dict = experiment_set.create_table()
    print(table_dict.keys())

    # try running the bars and whisker plots
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    fig, axes = plt.subplots(figsize=(8, 4))
    
    title = f"{args.stem} on {args.task}"
    if args.task == 'deprel':
        if args.source == 'selection_logit':
            title += " (head)"
        else:
            title += " (label)"
            
    axes.set_title(title, fontsize=20)
    
    axes.set_xticks(
        ticks=[i * STEP_SIZE for i in range(len(LANG_SEQ))],
    )

    axes.set_ylabel('ECE', fontsize=20)

    axes.set_xticklabels(
        labels=LANG_SEQ,
        fontsize=20
    )
    axes.tick_params(axis='y', labelsize=20)
    axes.set_frame_on(False)

    for midx, method in enumerate(args.method):
        samples = [table_dict[lang]['calibration'][method]['samples'] for lang in LANG_SEQ]
        deviation = - WIDTH / 2 + (WIDTH / len(args.method)) * midx
        
        axes.boxplot(
            x=samples,
            notch=True,
            sym="",
            positions=np.array([i * STEP_SIZE + deviation for i in range(len(LANG_SEQ))]),
            patch_artist=True,
            boxprops=dict(facecolor=COLORS[midx], color=COLORS[midx]),
            capprops=dict(color=COLORS[midx]),
            whiskerprops=dict(color=COLORS[midx]),
            flierprops=dict(color=COLORS[midx], markeredgecolor=COLORS[midx]),
            medianprops=dict(color=COLORS[midx]),
            manage_ticks=False,
        )
        axes.plot([], [], color=COLORS[midx], label=METHOD_LEGEND_MAP[method])

    # it would be better to also plot original ECE.
    [table_dict[lang] for lang in LANG_SEQ]

    axes.legend(fontsize=15)
        
    fig.tight_layout()
    fig.savefig(args.output_path)
        
if __name__ == '__main__':
    main()
