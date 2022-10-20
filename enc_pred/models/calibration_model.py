"""Calibration model to encompass Calibration
and metrics calculation.
"""
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from ..modules.logits_scaling.posthoc_calibration import ScalingModule, ScalingOutput
import torch
from typing import Text, Dict, List, Tuple, Optional, Any


@Model.register('calibration-model')
class CalibrationModel(Model):
    """We are not inheriting from Model subclass because
    we don't take vocabularies.
    """

    def __init__(
        self,
        scaling_module: ScalingModule,
        ori_metrics: Dict[Text, Metric],
        scaled_metrics: Dict[Text, Metric],
        vocab: Vocabulary,
        **extras
    ):
        """
        """
        super().__init__(vocab)

        self.scaling_module = scaling_module
        self._ori_metrics = ori_metrics
        self._scaled_metrics = scaled_metrics

    def forward(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        num_labels: Optional[torch.Tensor] = None,
        **extras,
    ) -> Dict[Text, torch.Tensor]:
        """
        """
        
        return_struct: ScalingOutput = self.scaling_module(
            input_=logits,
            label=labels
        )

        for key, val in self._ori_metrics.items():
            # currently use key name to identify relevant metrics
            if 'brier' in key:
                val(return_struct.original_logits, labels, num_labels=num_labels)
            else:
                val(return_struct.original_logits, labels)

        for key, val in self._scaled_metrics.items():
            if 'brier' in key:
                val(return_struct.logits, labels, num_labels=num_labels)
            else:
                val(return_struct.logits, labels)

        # calculate loss
        return {
            'loss': return_struct.loss,
            'logits': return_struct.logits
        }

    def get_metrics(self, reset: bool = False) -> Dict[Text, float]:
        """
        """
        metric_dict = {}
        for key, val in self._ori_metrics.items():
            for vk, vv in val.get_metric(reset).items():
                metric_dict[f'ori::{key}::{vk}'] = vv
        for key, val in self._scaled_metrics.items():
            for vk, vv in val.get_metric(reset).items():
                metric_dict[f'scaled::{key}::{vk}'] = vv

        return metric_dict