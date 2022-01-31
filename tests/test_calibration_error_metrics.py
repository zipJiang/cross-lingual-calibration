"""This function compare our implementation of metrics 
with easily observable ground truth and scoring results
given by ground truth third party packages.
"""
import unittest
import calibration as cal
import torch
from enc_pred.metrics import ExpectedCalibrationError
from typing import Text, List, Tuple, Dict, Union, Optional
import numpy as np
from tqdm import tqdm


class TestCalibrationMetrics(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        self.num_bin_list = [1, 5, 10]
        self.num_label_list = [10, 20, 100]
        self.ece_metric: ExpectedCalibrationError = ExpectedCalibrationError(
            num_bins=self.num_bin_list[0]
        )

        # TODO: adding more calibration scorers
        pass

    def test_calibration_error_on_expected_dataset(self):
        """
        """
        
        # generating test cases:
        def generate_perfectly_calibrated_instances(
            num_labels: int,
            num_instances: int,
            num_trial: int = 10,
        ):
            """This function generate required datset with perfect uniformed calibration prediction.
            Here the predictions are of marginal probability to reflect that ECE is not a proper scoring rule.
            """

            with torch.no_grad():
                prior_dist = torch.distributions.dirichlet.Dirichlet(
                    concentration=torch.ones((num_labels,), dtype=torch.float32)
                )
                make_up_dist = torch.distributions.Categorical(
                    probs=torch.ones((num_labels,), dtype=torch.float32) / num_labels
                )

                label_dist = prior_dist.sample((num_trial,))

                label_count = (label_dist * num_instances).long()

                # make up for labels
                total_labels = label_count.sum(dim=-1, keepdim=True)

                diff_shape = (num_instances - total_labels)

                supplements = torch.nn.functional.one_hot(
                    make_up_dist.sample(sample_shape=(num_trial, diff_shape.max().item() + 1)),
                    num_classes=num_labels
                )

                supplements = torch.where(
                    torch.arange(start=0, end=supplements.shape[1], step=1, dtype=torch.int64).view(1, -1, 1) < diff_shape.unsqueeze(-1),
                    supplements,
                    torch.zeros_like(supplements)
                )

                label_count = label_count + supplements.sum(1)

                # recalculate label_dist [num_trial, num_labels]
                label_dist = label_count.float() / num_instances

                # predictions = [num_trial, num_instances, num_labels]
                predictions = label_dist.unsqueeze(1).expand(-1, num_instances, -1)

                # [num_trial, , num_instances]
                selector = torch.arange(start=0, end=num_instances, step=1, dtype=torch.int64).view(1, 1, -1)
                gold_labels = torch.full(
                    size=(num_trial, num_labels, num_instances),
                    fill_value=-1,
                    dtype=torch.int64
                )

                filler = torch.arange(start=0, end=num_labels, step=1, dtype=torch.int64).view(1, -1, 1).expand(num_trial, -1, num_instances)
                gold_labels = torch.where(
                    selector < label_count.unsqueeze(-1),
                    filler,
                    gold_labels
                )

                gold_labels = gold_labels.flatten()

                gold_labels = gold_labels[gold_labels >= 0].reshape(num_trial, -1)

                assert gold_labels.shape[-1] == num_instances

            return {
                'predictions': predictions.log(),
                'gold_labels': gold_labels
            }

            
        # do a loop for testing calibration error correctness.
        for num_labels in self.num_label_list:
            for num_bins in self.num_bin_list:
                # This automatically reset the metrics
                self.ece_metric.num_bins = num_bins
                inputs = generate_perfectly_calibrated_instances(
                    num_labels=num_labels,
                    num_instances=20000,
                    num_trial=10
                )

                for predictions, gold_labels in zip(inputs['predictions'].split(1), inputs['gold_labels'].split(1)):
                    self.ece_metric(predictions=predictions.squeeze(0), gold_labels=gold_labels.squeeze(0))
                    metric_dict = self.ece_metric.get_metric()

                    for key, val in metric_dict.items():
                        self.assertAlmostEqual(val, 0., places=4, msg=f'{key}: {val} does not equal to 0., num_bins={num_bins}, num_labels={num_labels}')

    def test_random_calibration_error_against_gold(self):
        """Test against library implementation.
        """
        for _ in tqdm(range(20)):
            num_instance = 20000
            for num_labels in self.num_label_list:
                with torch.no_grad():
                    dirichlet_dist = torch.distributions.Dirichlet(concentration=torch.tensor([1.] * num_labels))
                    for num_bins in self.num_bin_list:
                        # sample a label distribution
                        label_dist = torch.distributions.Categorical(probs=dirichlet_dist.sample())
                        gold_labels = label_dist.sample(sample_shape=(num_instance,))
                        
                        # sample predictions
                        predictions = dirichlet_dist.sample(sample_shape=(num_instance,))
                        self.ece_metric.num_bins = num_bins
                        self.ece_metric(
                            predictions=predictions.log(),
                            gold_labels=gold_labels
                        )
                        self_ece = self.ece_metric.get_metric()['ECE']
                        ground_ece = cal.get_ece(
                            probs=predictions,
                            labels=gold_labels,
                            num_bins=num_bins
                        )

                        self.assertAlmostEqual(self_ece, ground_ece, places=4, msg=f'{num_instance}::{num_labels}')