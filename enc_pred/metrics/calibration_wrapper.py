from allennlp.training.metrics import Metric
from typing import Dict, Text, Optional
import torch
from overrides import overrides


@Metric.register("ud-calibration")
class CalibrationWrapper(Metric):
    def __init__(self, 
        label_metric: Metric,
        arc_metric: Metric
    ):
        """
        """
        self.label_metric = label_metric
        self.arc_metric = arc_metric

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
        self.arc_metric(predicted_indices_logits.flatten(0, 1), gold_indices.flatten(0, 1), mask=mask.flatten())
        self.label_metric(predicted_labels_logits.flatten(0, 1), gold_labels.flatten(0, 1), mask=mask.flatten())

    @overrides
    def get_metric(self, reset: bool = False) ->Dict[Text, float]:

        return_dict = {}

        for key, val in self.arc_metric.get_metric(reset).items():
            return_dict[f'arc-{key}'] = val

        for key, val in self.label_metric.get_metric(reset).items():
            return_dict[f'label-{key}'] = val

        return return_dict