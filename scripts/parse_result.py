"""
"""
from enc_pred.utils.spreadsheet import CalibrationExperiment, PredictionExperiment, ExperimentSet, ExperimentDataSizeGroup, ExperimentModelGroup, ExperimentTaskGroup
from 
import json
import argparse


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Parsing result from the run dirs.
        """
    )
    
    return parser.parse_args()

    
def main():
    args = parse_args()
    
    calibration_experiments = {}
    for method in [
        'beta-calibration',
        'gp-calibration',
        'histogram-binning',
        'temperature-scaling'
    ]:
        calibration_experiment = CalibrationExperiment.from_dir(f"runs/large-xlmr_deprel={method}=logit")
        calibration_experiments[method] = calibration_experiment

    experiment_set = ExperimentSet(
        prediction_experiment=PredictionExperiment.from_dir("runs/large-xlmr_deprel"),
        calibration_experiments=calibration_experiments
    )

    print(experiment_set.create_table())
    
    
if __name__ == '__main__':
    main()