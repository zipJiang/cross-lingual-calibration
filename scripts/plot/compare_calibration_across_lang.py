"""Plot model calibration result for
different sized models that changes
according to the datasize.
"""
from transformers import TemperatureLogitsWarper
from enc_pred.utils.spreadsheet import ExperimentSet, CalibrationExperiment, PredictionExperiment, ExperimentDataSizeGroup, ExperimentModelGroup
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.path as mpath

LANG_LIST = ['en', 'de', 'fr', 'es', 'ru', 'hi', 'ar', 'zh']
MODELS = ['large-xlmr', 'xlmr', 'mbert']

PARENT_DIR = "/brtx/604-nvme2/zpjiang/encode_predict/runs/"

DATA_SOURCE_MAP = {
    'full': '',
    'l': '-lr',
    'll': '-llr'
}

CLOSE = (86 / 255, 180 / 255, 233 / 255)
FAR = (233 / 255, 159 / 255, 0 / 255)


def parse_args():
    parser = argparse.ArgumentParser(
        """Plotting model calibration across different datasize
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    fig, axes = plt.subplots(figsize=(10, 5))
    
    for model_name in MODELS:
        predict_dir = f"{model_name}{DATA_SOURCE_MAP[args.data_source]}_{args.task}"
        
        prediction_experiment = PredictionExperiment.from_dir(
            os.path.join(PARENT_DIR, predict_dir)
        )
        
        # Here we presume that we are not working on head.
        calibration_experiment = CalibrationExperiment.from_dir(
            os.path.join(PARENT_DIR, f"{predict_dir}=temperature-scaling=logit")
        )

        experiment_set = ExperimentSet(
            prediction_experiment=prediction_experiment,
            calibration_experiments={
                'temperature-scaling': calibration_experiment
            }
        )
        
        data = experiment_set.create_table()
        ece = [data[lang]['calibration']['original']['mean'] for lang in LANG_LIST]
        
        circle = mpath.Path.unit_circle()
        if model_name.startswith('large'):
            marker_size = 15
        else:
            marker_size = 5
            
        if model_name.endswith('r'):
            color = CLOSE
        else:
            color = FAR
            
        axes.plot(ece, f'--', marker=circle, markersize=marker_size, color=color, alpha=.6, label=model_name)

    axes.set_title(args.task, fontsize=20)
    axes.set_xticks(list(range(8)))
    axes.set_xticklabels(LANG_LIST, fontsize=20)

    axes.tick_params(axis='y', labelsize=20)
    axes.set_frame_on(False)

    axes.set_ylabel('ECE', fontsize=20)
    axes.legend(prop={'size': 20})
        
    fig.tight_layout()
    fig.savefig(args.output_path)
    
    
if __name__ == '__main__':
    main()
