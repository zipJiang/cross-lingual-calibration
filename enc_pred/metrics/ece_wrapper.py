from allennlp.training.metrics import Metric
from typing import Dict, Text, Optional
import torch
from overrides import overrides

from .calibration_error import ExpectedCalibrationError


@Metric.register("ud-ece")
class ECEWrapper(Metric):
    def __init__(self, num_bins: int = 10):
        """
        """
        self.label_calibration = ExpectedCalibrationError(num_bins=num_bins)
        self.arc_calibration = ExpectedCalibrationError(num_bins=num_bins)

    @overrides
    def __call__(
        self,
        predicted_indices_logits: torch.Tensor,
        predicted_labels_logits: torch.Tensor,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        """
        self.arc_calibration(predicted_indices_logits.flatten(0, 1), gold_indices.flatten(0, 1), mask=mask.flatten())
        self.label_calibration(predicted_labels_logits.flatten(0, 1), gold_labels.flatten(0, 1), mask=mask.flatten())

    @overrides
    def get_metric(self, reset: bool = False) ->Dict[Text, float]:

        return {
            'arc-ECE': self.arc_calibration.get_metric(reset)['ECE'],
            'label-ECE': self.label_calibration.get_metric(reset)['ECE']
        }