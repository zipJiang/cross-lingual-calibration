"""Linear Classification Head is just a linear predictor
that transform the representation once.
"""
from .prediction_head import PredictionHead
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common import Lazy
import torch
from typing import Text, Dict, Tuple, Optional, List, Any
from overrides import overrides


@PredictionHead.register('linear-classification-head')
@PredictionHead.register('lazy-linear-classification-head', constructor='lazy_construct')
class LinearClassificationHead(PredictionHead):
    def __init__(self,
        vocabulary: Vocabulary,
        input_dim: int,
        label_namespace: Text = 'labels',
        with_bias: bool = False
    ):
        """
        """
        super(
            vocabulary=vocabulary,
            input_dim=input_dim
        ).__init__()

        self.with_bias = with_bias
        self.label_namespace = label_namespace

        self.linear = torch.nn.Linear(
            in_features=input_dim,
            out_features=vocabulary.get_vocab_size(namespace=self.label_namespace),
            bias=self.with_bias)

    @classmethod
    def lazy_construct(cls,
        vocabulary: Vocabulary,
        encoder: Seq2SeqEncoder,
        label_namespace: Text = 'labels',
        with_bias: Optional[bool] = False,
        **extras
    ) -> "PredictionHead":
        """
        """
        return cls(
            vocabulary=vocabulary,
            input_dim=encoder.get_output_dim(),
            with_bias=with_bias
        )

    @overrides
    def forward(
        self,
        span_repr: torch.Tensor,
        span_mask: torch.Tensor,
        **extra,
    ) -> Dict[Text, torch.Tensor]:
        """
        """
        batch_size, num_spans = span_repr.size()
        span_repr = span_repr.flatten(0, 1)

        # linear transform
        pred_logits = self.linear(span_repr)

        return {
            'logits': pred_logits.view(batch_size, num_spans, -1),
            'span_mask': span_mask,
            'num_spans': torch.sum(span_mask.int(), dim=-1)
        }