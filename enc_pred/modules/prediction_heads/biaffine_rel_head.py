"""This biaffine rel_head is used to predict the universal dependency labels
"""
from .prediction_head import PredictionHead
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common import Lazy
from allennlp.nn.activations import Activation
import torch
from typing import Text, Dict, Tuple, Optional, List, Any
from overrides import overrides


@PredictionHead.register('biaffine')
@PredictionHead.register('lazy-biaffine')
class BiaffineRelHead(PredictionHead):
    """
    """
    def __init__(self,
        vocabulary: Vocabulary,
        input_dim: int,
        hidden_dim: int,
        activation: Activation,
        label_namespace: Text = 'labels',
        with_bias: Optional[bool] = False,

    ):
        """
        """
        super(
            vocabulary=vocabulary,
            input_dim=input_dim
        ).__init__()

        self.with_bias = with_bias
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.label_namespace = label_namespace

        self.head_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=vocabulary.get_vocab_size(namespace=self.label_namespace),
                bias=self.with_bias),
            self.activation
        )

        self.dep_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=vocabulary.get_vocab_size(namespace=self.label_namespace),
                bias=self.with_bias),
            self.activation
        )

        # also we need to register two bi_affine layers
        self.W_arc = torch.nn.Parameter(torch.tensor(self.get_output_dim(), self.hidden_dim, self.hidden_dim, torch.float32))
        self.b_arc = torch.nn.Linear(2 * self.hidden_dim, self.get_output_dim(), bias=True)
        self.class_prior =  torch.nn.Parameter(torch.tensor(self.get_output_dim(), dtype=torch.float32))

    @classmethod
    def lazy_construct(cls,
        vocabulary: Vocabulary,
        hidden_dim: int,
        activation: Activation,
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
            hidden_dim=hidden_dim,
            activation=activation,
            with_bias=with_bias
        )

    def forward(
        self,
        span_repr: torch.Tensor,
        span_mask: torch.Tensor,
        parent_ids: torch.Tensor,
        **extras
    ) -> Dict[Text, torch.Tensor]:
        """
        parent_ids: [batch_size, num_spans]
        """
        
        batch_size, num_spans, feature_dim = span_repr.size()
        span_repr = span_repr.flatten(0, 1)
        parent_repr = torch.gather(
            span_repr,
            index=parent_ids.unsqueeze(-1),
        )
        parent_repr = parent_repr.glatten(0, 1)

        head_repr = self.head_mlp(parent_repr)
        dep_repr = self.dep_mlp(span_repr)

        single_side = self.b_arc(torch.cat(head_repr, dep_repr, dim=-1))  # [batch_size * num_spans, num_classes]
        left_prod = torch.matmul(head_repr.view(-1, 1, 1, feature_dim), self.W_arc.unsqueeze(0))
        right_prod = torch.matmul(left_prod, dep_repr.view(-1, 1, feature_dim, 1)).view(-1, self.get_output_dim())  # [batch_size * num_spans, num_classes]

        logits = right_prod + single_side + self.class_prior.unsqueeze(0)

        return {
            'logits': logits.view(batch_size, num_spans, feature_dim),
            'span_mask': span_mask,
            'num_spans': torch.sum(span_mask.int(), dim=-1)
        }