"""Enc sent pred model used
to predict a label on sentence/sentence-pair.
"""
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from sklearn.decomposition import sparse_encode
from ..modules.prediction_heads.prediction_head import PredictionHead
from allennlp.nn.initializers import InitializerApplicator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import Metric
from allennlp.common.lazy import Lazy

import torch
from typing import Text, Dict, List, Tuple, Optional, Any
from overrides import overrides

from ..modules.prediction_heads.prediction_head import PredictionHead


@Model.register('enc-sentpred')
@Model.register('enc-sentpred-lazy', constructor='from_lazy_object')
class EncSentPredModel(Model):
    def __init__(
        self,
        word_embedding: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        prediction_head: PredictionHead,
        metrics: Dict[Text,Metric],
        vocab: Vocabulary,
        initializer: Optional[InitializerApplicator] = None
    ):
        """Encode-predict model for the sentence prediction model.
        """
        super().__init__(vocab)
        self.embedding = word_embedding
        self.seq2_vec_encoder = seq2vec_encoder
        self.prediction_head = prediction_head
        self.metrics = metrics

        if initializer is not None:
            initializer(self)

    @classmethod
    def from_lazy_object(
        cls,
        word_embedding: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        prediction_head: Lazy[PredictionHead],
        metrics: Dict[Text, Metric],
        vocab: Vocabulary,
        initializer: Optional[InitializerApplicator] = None
    ) -> "EncSentPredModel":
        """Enc Sentence prediction model from lazy object because input dim
        could not be known in advance.
        """
        prediction_head = prediction_head.construct(
            vocabulary=vocab,
            input_dim=seq2vec_encoder.get_output_dim()
        )

        return cls(
            word_embedding,
            seq2vec_encoder,
            prediction_head,
            metrics,
            vocab,
            initializer
        )

    def forward(
        self,
        tokens: Dict[Text, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        *args, **kwargs
    ) -> Dict[Text, torch.Tensor]:
        """Normal sentence classification implementation
        """

        token_vec = self.embedding(tokens)
        sentence_repr = self.seq2_vec_encoder(token_vec, mask=tokens['pieces']['mask'])

        prediction = self.prediction_head(sentence_repr)
        logits = prediction['logits']

        if labels is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            return_dict = {
                'loss': loss_func(logits, labels),
                'logits': logits
            }

            # call and calculate metrics
            for metric in self.metrics.values():
                metric(
                    predictions=logits.view(-1, logits.shape[-1]),
                    gold_labels=labels.flatten()
                )

        else:
            return_dict = {
                'logits': logits
            }

        return return_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[Text, float]:
        """
        """
        metrics = {
            f'{key_met}::{key}': val for key_met, val_met in self.metrics.items()
            for key, val in val_met.get_metric(reset).items()
        }

        return metrics
