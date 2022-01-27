"""Wrapping over attachment scores for uniformity
"""
from allennlp.training.metrics import Metric
from allennlp.training.metrics.attachment_scores import AttachmentScores
from typing import Dict, Text, Optional
import torch
from overrides import overrides


@Metric.register('attachment-logits')
class AttachmentWrapper(Metric):
    """
    """
    def __init__(self):
        self.base_metric = AttachmentScores()

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
        self.base_metric(
            predicted_indices=predicted_indices_logits.argmax(dim=-1),
            predicted_labels=predicted_labels_logits.argmax(dim=-1),
            gold_indices=gold_indices,
            gold_labels=gold_labels,
            mask=mask
        )

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[Text, float]:
        """
        """
        return self.base_metric.get_metric(reset)