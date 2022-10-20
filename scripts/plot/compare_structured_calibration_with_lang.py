"""Plot model calibration result for
different sized models that changes
according to the datasize. (Notice that
since original is read across different model
and cannot be parsed, we here manually read the data
file.)
"""
from transformers import TemperatureLogitsWarper
from enc_pred.utils.spreadsheet import ExperimentSet, CalibrationExperiment, PredictionExperiment, ExperimentDataSizeGroup, ExperimentModelGroup
import argparse
import os
import json
import matplotlib.pyplot as plt
import matplotlib.path as mpath

from scripts.plot.compare_calibration_methods import METHOD_LEGEND_MAP


PARENT_DIR = '/brtx/604-nvme2/zpjiang/encode_predict/seqtag-runs/'

LANG_LIST = ['en', 'de', 'fr', 'es', 'ru', 'hi', 'ar', 'zh']
MODELS = ['large-xlmr', 'xlmr']

METHOD_MAP = {
    'temperature-scaling': 'TS',
    'gp-calibration': 'GPcalib'
}

TASK_MAP = {
    'wikiann': 'ner',
    'xnli': 'xnli',
}

DATA_SOURCE_MAP = {
    'full': '',
    'l': '-lr',
    'll': '-llr'
}

CLOSE = (86 / 255, 180 / 255, 233 / 255)
FAR = (233 / 255, 159 / 255, 0 / 255)

USED_LABEL = set()


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Plotting model performance w.r.t. language
        with a stem
        """
    )
    
    parser.add_argument(
        '--task', action='store', dest='task',
        type=str, required=True, help='Which task to run on.'
    )
    parser.add_argument(
        '--output_path', action='store', dest='output_path',
        type=str, required=True, help='Where to write the data.'
    )
    
    parser.add_argument(
        '--data_source', action='store', dest='data_source',
        type=str, required=False, default='full'
    )
    
    parser.add_argument(
        '--step', action='store_true', dest='step',
        required=True, help='Whether to store step files.'
    )
    
    # parser.add_argument(
    #     '--model_stem', action='store', dest='model_stem',
    #     type=str, required=True, help='Which base model to check.'
    # )
    
    return parser.parse_args()

    
def main():

    args = parse_args()

    fig, axes = plt.subplots(figsize=(10, 5))
    
    # for model_stem in MODELS:
    #     for suffix in ['', '-crf']:

    #         calibration_dirname = f"{PARENT_DIR}{model_stem}{DATA_SOURCE_MAP[args.data_source]}-{args.task}{suffix}=temperature-scaling=logit/1/eval"
            
    #         result = []
    #         for lang in LANG_LIST:
    #             with open(os.path.join(calibration_dirname, f"{lang}.json"), 'r', encoding='utf-8') as file_:
    #                 val = json.load(file_)['ori::ece::ECE']
    #                 result.append(val)
                    
    #         circle = mpath.Path.unit_circle()
            
    #         if model_stem.startswith('large'):
    #             marker_size = 15
    #         else:
    #             marker_size = 5

    #         axes.plot(result, marker=circle, markersize=marker_size, alpha=.6, color=CLOSE if suffix == '' else FAR, label=f"{model_stem}{suffix}")
    axes.set_title(TASK_MAP[args.task], fontsize=20)
    axes.set_xticks(list(range(8)))
    axes.set_xticklabels(LANG_LIST, fontsize=20)

    axes.tick_params(axis='y', labelsize=20)
    axes.set_frame_on(False)

    axes.set_ylabel('ECE')
    axes.set_yticks([0.05, 0.10, 0.15])
    axes.set_ylim([0.0, 0.19])
    
    _get_step_filename = lambda x: ''.join([*args.output_path.split('.')[:-1], f'-step-{x}.', args.output_path.split('.')[-1]])
        
    # now for the second loop try to plot calibration result
    for step, method in enumerate(['temperature-scaling', 'gp-calibration']):
        for model_stem in MODELS:
            for suffix in ['', '-crf']:
                calibration_dirname = f"{PARENT_DIR}{model_stem}{DATA_SOURCE_MAP[args.data_source]}-{args.task}{suffix}={method}=logit/"
                
                result = []
                for lang in LANG_LIST:
                    stack = []
                    for i in range(1, 11):
                        with open(os.path.join(calibration_dirname, str(i), "eval", f"{lang}.json"), 'r', encoding='utf-8') as file_:
                            stack.append(json.load(file_)['scaled::ece::ECE'])
                            
                    result.append(sum(stack) / len(stack))
                
                
                if model_stem.startswith('large'):
                    marker_size = 15
                else:
                    marker_size = 5

                label_str = METHOD_LEGEND_MAP[method] + suffix
                if label_str not in USED_LABEL:
                    USED_LABEL.add(label_str)
                else:
                    label_str = None
                    
                axes.plot(result, '--', marker='s' if suffix == '' else 'p', markersize=marker_size, alpha=.6, color=CLOSE if method == 'temperature-scaling' else FAR, label=label_str)
        axes.legend(prop={'size': 20})
        fig.savefig(_get_step_filename(step))
    
    fig.savefig(args.output_path)
    
    
if __name__ == '__main__':
    main()