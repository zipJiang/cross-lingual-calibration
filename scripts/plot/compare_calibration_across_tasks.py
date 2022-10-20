"""This scripts do the plot similar to
the original plots in figure-1, and try
 to show how model calibration is affected
 by training data-source and task.
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

CLOSE = (86 / 255, 180 / 255, 233 / 255)
FAR = (233 / 255, 159 / 255, 0 / 255)


RUN_DIR = '/brtx/604-nvme2/zpjiang/encode_predict/runs/'
TASK_LIST = ['pos_tags', 'deprel', 'ner', 'xnli']
DATA_SOURCE_LIST = ['', '-lr', '-llr']
WIDTH=2
STEP_SIZE=4

TASK_GROUP_NAMES = ['POS', 'UDP-l', 'UDP-h', 'NER', 'XNLI']
TASK_GROUP_NAME_MAP = {
    'ner-logit': 'NER',
    'pos_tags-logit': 'POS',
    'deprel-logit': 'UDP-l',
    'deprel-selection_logit': 'UDP-h',
    'xnli-logit': 'XNLI'
}


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
        default='temperature-scaling'
    )
    parser.add_argument(
        '--model', action='store', dest='model', type=str, required=True,
        help='Which model to load.'
    )
    parser.add_argument(
        '--output_path', action='store', dest='output_path',
        type=str, required=True, help='Where write the output.'
    )
    parser.add_argument(
        '--lang', action='store', dest='lang',
        type=str, required=False, default='en',
        choices=['en', 'es', 'fr', 'de', 'ru', 'hi', 'ar', 'zh']
    )
    
    return parser.parse_args()


def main():
    """
    """
    args = parse_args()
    

    task_group = {}
    for task in TASK_LIST:
        source_list = ['logit'] if task != 'deprel' else ['logit', 'selection_logit']
        for ss in source_list:
            data_size_group = {}
            for data_source in DATA_SOURCE_LIST:
                stem = f'{args.model}{data_source}_{task}'

                prediction_experiment = PredictionExperiment.from_dir(
                    dirname=os.path.join(RUN_DIR, stem)
                )
                
                calibration_experiments = {
                    args.method: CalibrationExperiment.from_dir(
                        os.path.join(RUN_DIR, f"{stem}={args.method}={ss}")
                    )
                }
                
                experiment_set = ExperimentSet(
                    prediction_experiment=prediction_experiment,
                    calibration_experiments=calibration_experiments
                )
                
                data_size_group[data_source] = experiment_set
                
            task_group[TASK_GROUP_NAME_MAP[f"{task}-{ss}"]] = ExperimentDataSizeGroup(
                sub_groups=data_size_group
            )

    task_groups = ExperimentTaskGroup(
        sub_groups=task_group
    )
    
    task_group_dict = task_groups.create_table()
    
    fig, axes = plt.subplots(figsize=(8, 4))
    
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    fig, axes = plt.subplots(figsize=(8, 4))
    
    title = f"{args.model} on {args.lang}"
            
    axes.set_title(title, fontsize=20)
    
    axes.set_xticks(
        ticks=[i * STEP_SIZE for i in range(len(TASK_GROUP_NAMES))],
    )
    axes.set_xticklabels(
        labels=TASK_GROUP_NAMES,
        fontsize=20
    )
    axes.tick_params(axis='y', labelsize=20)
    axes.set_frame_on(False)
    
    axes.set_ylabel('ECE')
    
    base_ruler = np.array([i * STEP_SIZE for i in range(len(TASK_GROUP_NAMES))])
    for didx, data_size in enumerate(DATA_SOURCE_LIST):
        
        deviation = WIDTH / len(DATA_SOURCE_LIST) * didx - WIDTH / 2
        
        vals = np.array([task_group_dict[task_name][data_size][args.lang]['calibration'][args.method]['mean'] for task_name in TASK_GROUP_NAMES])
        ori_vals = np.array([task_group_dict[task_name][data_size][args.lang]['calibration']['original']['mean'] for task_name in TASK_GROUP_NAMES])
        
        axes.bar(base_ruler + deviation, vals, width=WIDTH / len(DATA_SOURCE_LIST) * .9, color=CLOSE, label='after' if didx == 0 else None, align='center')
        axes.bar(base_ruler + deviation, (ori_vals - vals), width=WIDTH / len(DATA_SOURCE_LIST) * .9, color=FAR, bottom=vals, label='before' if didx == 0 else None, align='center')

    handles, labels = axes.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: 1 if x[0] == 'before' else 2, reverse=False))
    
    axes.legend(handles, labels, prop={"size": 20})
    
    fig.tight_layout()
    fig.savefig(args.output_path)
    
    
if __name__ == '__main__':
    main()