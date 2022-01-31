"""EncPredict models
"""
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn.initializers import InitializerApplicator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import Metric
from allennlp.common.lazy import Lazy
import torch
from typing import Text, Dict, List, Tuple, Optional, Any
from overrides import overrides

from ..modules.prediction_heads.prediction_head import PredictionHead


torch.autograd.set_detect_anomaly(True)


@Model.register('enc-predict')
@Model.register('enc-predict-lazy', constructor='from_lazy_object')
class EncPredictModel(Model):
    """
    """
    def __init__(
        self,
        word_embedding: TextFieldEmbedder,
        span_extractor: SpanExtractor,
        prediction_head: PredictionHead,
        metrics: Dict[Text, Metric],
        vocab: Vocabulary,
        initializer: Optional[InitializerApplicator] = None,
    ):
        """
        """
        super().__init__(vocab)
        self.embedding = word_embedding
        self.span_extractor = span_extractor
        self.metrics = metrics
        self.prediction_head = prediction_head

        if initializer is not None:
            initializer(self)

    @classmethod
    def from_lazy_object(
        cls,
        word_embedding: TextFieldEmbedder,
        span_extractor: Lazy[SpanExtractor],
        prediction_head: Lazy[PredictionHead],
        metrics: Dict[Text, Metric],
        vocab: Vocabulary,
        initializer: Optional[InitializerApplicator] = None,
        **extras
    ) -> "EncPredictModel":
        """
        """
        span_extractor = span_extractor.construct(
            input_dim=word_embedding.get_output_dim()
        )
        prediction_head = prediction_head.construct(
            vocabulary=vocab,
            input_dim=span_extractor.get_output_dim()
        )

        return cls(
            word_embedding=word_embedding,
            span_extractor=span_extractor,
            prediction_head=prediction_head,
            metrics=metrics,
            vocab=vocab,
            initializer=initializer
        )

    def forward(
        self,
        tokens: Dict[Text, torch.Tensor],
        spans: torch.Tensor,
        parent_ids: Optional[torch.Tensor] = None,
        parent_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **extras
    ) -> Dict[Text, torch.Tensor]:
        """
        """
        token_vec = self.embedding(tokens)
        # generate a span_representation mask
        span_mask = self._get_span_repr_mask(spans)

        if parent_mask is not None:
            span_mask = torch.logical_and(span_mask, parent_mask)

        # [batch_size, num_spans, feature_dim]
        span_repr = self.span_extractor(
            sequence_tensor=token_vec,
            sequence_mask=tokens['pieces']['mask'],
            span_indices=spans,
            span_indices_mask=span_mask
        )

        predictions = self.prediction_head(
            span_repr=span_repr,
            span_mask=span_mask,
            parent_ids=parent_ids,
        )

        # comparing against labels
        logits = predictions['logits']
        selection_logits = None
        
        # collect labels according to task.
        # TODO: move the flatten / update metric inside self-calibration metrics
        # TODO: move the detachment int othe metric

        if labels is not None:
            for metric in self.metrics.values():
                if parent_ids is not None:
                    metric(
                        predicted_indices_logits=predictions['selection_logits'],
                        predicted_labels_logits=predictions['logits'],
                        gold_indices=parent_ids,
                        gold_labels=labels,
                        mask=span_mask
                    )
                else:
                    metric(
                        predictions=logits.view(-1, logits.shape[-1]),
                        gold_labels=labels.flatten(),
                        mask=span_mask.flatten()
                    )

            loss_func = torch.nn.CrossEntropyLoss()
            labels[~span_mask] = -100

            # maybe also calculate selection loss
            if parent_ids is not None:
                selection_logits = predictions['selection_logits']
                # parent_ids[~span_mask] = -100
                parent_labels = parent_ids.masked_scatter(
                    mask=~span_mask,
                    source=torch.ones_like(parent_ids)
                )
                parent_loss = loss_func(selection_logits.flatten(0, 1), parent_labels.flatten(0, 1))
            else:
                parent_loss = 0.
            return_dict = {
                'loss': loss_func(logits.flatten(0, 1), labels.flatten(0, 1)) + parent_loss
            }
        else:
            return_dict = {}

        return_dict.update({
            # 'loss': loss_func(logits.flatten(0, 1), labels.flatten(0, 1)) + parent_loss,
            'logits': logits,
            'selection_logits': selection_logits
        })

        return return_dict


    def _get_span_repr_mask(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        """
        return boundaries[..., 0] != -1

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[Text, float]:
        """
        """
        metrics = {
            f'{key_met}::{key}': val for key_met, val_met in self.metrics.items()
            for key, val in val_met.get_metric(reset).items()
        }

        return metrics