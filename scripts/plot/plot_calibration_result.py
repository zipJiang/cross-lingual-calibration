"""This function will plot calibration reliability plot
given the input calibration file and a transfering model.
"""
import os
from allennlp.nn.util import move_to_device
from allennlp.models.model import Model
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from enc_pred.data_readers.calibration_readers import CalibrationReader
from enc_pred.metrics.calibration_error import ExpectedCalibrationError
import argparse
import torch


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Generating reliability plot given
        the original input files and new files.
        """
    )
    parser.add_argument(
        '--num_bins', action='store', dest='num_bins',
        type=int, required=True, help='Number of bins to use'
    )
    parser.add_argument(
        '--input_path', action='store', dest='input_path',
        type=str, required=True, help='Where to load input data.'
    )
    parser.add_argument(
        '--archive_path', action='store', dest='archive_path',
        type=str, required=False, default=None, help='Where to load archive data.'
    )

    parser.add_argument(
        '--is_head_pred', action='store_true', dest='is_head_pred',
        required=False, help='Whether to use "selection_logit".'
    )

    parser.add_argument(
        '--result_dir', action='store', dest='result_dir',
        type=str, required=True, help='Where to write the plot result.'
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.archive_path is not None:
        model = Model.from_archive(
            archive_file=args.archive_path
        )
        model.to(device=0)

        for pname, param in model.named_parameters():
            print(pname, param)
    else:
        model = None

    data_reader = CalibrationReader(
        is_prediction=False,
        logits_key='selection_logit' if args.is_head_pred else 'logit',
        labels_key='selection_label' if args.is_head_pred else 'label'
    )

    dataloader = SimpleDataLoader(
        instances=data_reader.read(args.input_path),
        batch_size=4096,
        shuffle=False
    )

    ori_metric: ExpectedCalibrationError = ExpectedCalibrationError(num_bins=args.num_bins)
    scaled_metric: ExpectedCalibrationError = ExpectedCalibrationError(num_bins=args.num_bins)

    for batch in dataloader:
        move_to_device(batch, 0)
        print(batch['labels'])
        ori_metric(
            predictions=batch['logits'],
            gold_labels=batch['labels']
        )

        if model is not None:
            with torch.no_grad():
                scaled_logit = model(**batch)['logits']
                scaled_metric(
                    predictions=scaled_logit,
                    gold_labels=batch['labels']
                )

    os.makedirs(args.result_dir, exist_ok=True)
    ori_fig = ori_metric.plot_reliability_diag()
    ori_fig.savefig(os.path.join(args.result_dir, 'ori.svg'))

    confidence_hist = ori_metric.plot_confidence_hist()
    confidence_hist.savefig(os.path.join(args.result_dir, 'ori-confidence.svg'))
    print(f"ori: {ori_metric.get_metric()}")

    if model is not None:
        scaled_fig = scaled_metric.plot_reliability_diag()
        scaled_fig.savefig(os.path.join(args.result_dir, 'scaled.svg'))

        # also plot confidence distribution
        scaled_confidence_hist = scaled_metric.plot_confidence_hist()
        scaled_confidence_hist.savefig(os.path.join(args.result_dir, 'scaled-confidence.svg'))
        print(f"scaled: {scaled_metric.get_metric()}")


if __name__ == '__main__':
    main()
