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
        super().__init__(
            vocabulary=vocabulary,
            input_dim=input_dim
        )

        self.with_bias = with_bias
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.label_namespace = label_namespace

        self.head_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=self.hidden_dim,
                bias=self.with_bias),
            self.activation
        )
        self.dep_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=self.hidden_dim,
                bias=self.with_bias),
            self.activation
        )
        self.head_label_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=self.hidden_dim,
                bias=self.with_bias),
            self.activation
        )
        self.dep_label_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=self.hidden_dim,
                bias=self.with_bias),
            self.activation
        )

        # also we need to register two bi_affine layers
        self.W_arc = torch.nn.Bilinear(
            in1_features=self.hidden_dim,
            in2_features=self.hidden_dim,
            out_features=self.vocabulary.get_vocab_size(namespace=self.label_namespace),
            bias=True
        )
        self.b_arc = torch.nn.Linear(2 * self.hidden_dim, self.vocabulary.get_vocab_size(namespace=self.label_namespace), bias=False)

        # apply arc_head prediction
        self.W_arc_transform = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.b_arc_transform = torch.nn.Linear(self.hidden_dim, 1, bias=False)

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

        # rationalizing parent_ids
        parent_ids[~span_mask] = 0

        parent_repr = torch.gather(
            span_repr,
            index=parent_ids.unsqueeze(-1).expand(-1, -1, feature_dim),
            dim=1
        )

        span_repr = span_repr.flatten(0, 1)
        parent_repr = parent_repr.flatten(0, 1)

        head_label_repr = self.head_label_mlp(parent_repr)
        dep_label_repr = self.dep_label_mlp(span_repr)

        single_side = self.b_arc(torch.cat((head_label_repr, dep_label_repr), dim=-1))  # [batch_size * num_spans, num_classes]
        bi_affine = self.W_arc(head_label_repr, dep_label_repr)

        logits = bi_affine + single_side

        # also calculate head selection [batch_size * num_spans, hidden_dim]
        hs_arc_repr = self.head_mlp(span_repr).view(batch_size, num_spans, -1)
        ds_arc_repr = self.dep_mlp(span_repr).view(batch_size, num_spans, -1)

        # [batch_size, num_spans, num_spans]
        mult_ = torch.bmm(self.W_arc_transform(ds_arc_repr), hs_arc_repr.transpose(1, 2))
        add_ = self.b_arc_transform(hs_arc_repr).transpose(1, 2)

        selection_logits = mult_ + add_

        return {
            'logits': logits.view(batch_size, num_spans, self.vocabulary.get_vocab_size(namespace=self.label_namespace)),
            'selection_logits': selection_logits,
            'span_mask': span_mask,
            'num_spans': torch.sum(span_mask.int(), dim=-1)
        }