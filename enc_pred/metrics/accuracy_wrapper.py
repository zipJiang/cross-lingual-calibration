"""This metrics is wrapper over categorical_accuracy in allennlp,
we try to return a dictionary that unifies code.
"""
from allennlp.training.metrics import Metric
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from typing import Dict, Text, Optional
import torch
from overrides import overrides


@Metric.register('dict-categorical')
class DictCategoricalAccuracy(Metric):
    """
    """
    def __init__(self):
        """
        """
        self.base_metric = CategoricalAccuracy()

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[Text, float]:
        """A wrapper
        """
        return {'accuracy': self.base_metric.get_metric(reset)}

    @overrides
    def __call__(self, predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        """
        self.base_metric(predictions, gold_labels, mask)