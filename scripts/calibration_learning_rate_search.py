"""This script will run hyperparameter search with calibration
allennlp training.
"""
import optuna
from optuna.integration import AllenNLPExecutor
from optuna.integration.allennlp import dump_best_config
from optuna.pruners import HyperbandPruner
from packaging import version
import allennlp
from typing import Text, List, Tuple, Dict, Iterable, Any, Optional
from functools import partial
import argparse
import shutil


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Search hyperparameter for learning_rate.
        """
    )
    parser.add_argument(
        '--serialization_dir', action='store', dest='serialization_dir',
        type=str, required=True, help='The parent directory of all hyper-search dirs.'
    )
    parser.add_argument(
        '--config_path', action='store', dest='config_path', type=str,
        required=True, help='Where to load the .jsonnet config.'
    )
    parser.add_argument(
        '--db_stem', action='store', dest='db_stem',
        type=str, required=True
    )
    parser.add_argument(
        '--config_dump_path', action='store', dest='config_dump_path',
        type=str, required=True, help='Where to dump the best configuration.'
    )

    return parser.parse_args()


def objective(
    trial: optuna.trial.Trial,
    config_path: Text,
    serialization_dir: Text):
    """
    """
    trial.suggest_float('LEARNING_RATE', low=5e-2, high=.5, log=True)

    executor = AllenNLPExecutor(
        trial,
        config_file=config_path,
        include_package='enc_pred',
        serialization_dir=f'{serialization_dir}/{trial.number}',
        metrics='best_validation_scaled::ece::ECE'
    )

    return executor.run()


def main():
    """
    """
    args = parse_args()

    study = optuna.create_study(
        direction='minimize',
        storage=f"sqlite:////tmp/{args.db_stem}.db",
        pruner=HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        partial(
            objective,
            config_path=args.config_path,
            serialization_dir=args.serialization_dir
        ),
        n_jobs=1,
        n_trials=10,
        timeout=1800
    )
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f" Value: {trial.value}")
    print(f" Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    dump_best_config(
        args.config_path,
        args.config_dump_path,
        study
    )

    shutil.rmtree(args.serialization_dir)


if __name__ == '__main__':
    main()