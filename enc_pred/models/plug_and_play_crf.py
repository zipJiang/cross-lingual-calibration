"""This model is used to train a CRF
model for BIO-tagging. The design of the
model is intentionally making comparison
w/o CRF module easy.
"""
from allennlp.models import Model
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn.initializers import InitializerApplicator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import Metric
from allennlp.common.lazy import Lazy
from allennlp.modules.conditional_random_field import ConditionalRandomField
from ..modules.crf.smooth_crf import SmoothCRF
from typer import Option
from ..modules.prediction_heads import PredictionHead
import torch
from typing import Text, Dict, List, Tuple, Optional, Any
from overrides import overrides


@Model.register('ppcrf', constructor='from_lazy_object')
class PlugAndPlayCRF(Model):
    """This model is compatible with NER and POS, where there is no further dependency
    between label and parents.
    """
    def __init__(
        self,
        word_embedding: TextFieldEmbedder,
        prediction_head: PredictionHead,
        metrics: Dict[Text, Metric],
        vocab: Vocabulary,
        crf: Optional[SmoothCRF] = None,
        initializer: Optional[InitializerApplicator] = None,
        token_namespace: Text = 'pieces',
        label_namespace: Text = 'labels'
    ):
        """In this model we are not predicting over spans,
        so that the label will be of bio-type (over sequence).
        """
        super().__init__(vocab)
        self.embedding = word_embedding
        self.metrics = metrics
        self.prediction_head = prediction_head
        
        self.crf = crf

        if initializer is not None:
            initializer(self)

        self._token_namespace = token_namespace
        self._label_namespace = label_namespace
        
    @classmethod
    def from_lazy_object(
        cls,
        word_embedding: TextFieldEmbedder,
        prediction_head: Lazy[PredictionHead],
        metrics: Dict[Text, Metric],
        vocab: Vocabulary,
        with_crf: bool = False,
        initializer: Optional[InitializerApplicator] = None,
        token_namespace: Text = 'pieces',
        label_namespace: Text = 'labels'
    ):
        """
        """
        prediction_head = prediction_head.construct(
            label_namespace=label_namespace,
            input_dim=word_embedding.get_output_dim(),
            vocabulary=vocab
        )
        
        if with_crf:
            # create the crf module
            crf = SmoothCRF(
                num_tags=vocab.get_vocab_size(label_namespace),
                include_start_end_transitions=True
            )
        else:
            crf = None

        return cls(
            word_embedding=word_embedding,
            prediction_head=prediction_head,
            metrics=metrics,
            vocab=vocab,
            crf=crf,
            initializer=initializer,
            token_namespace=token_namespace,
            label_namespace=label_namespace
        )
        
    def forward(
        self,
        tokens: Dict[Text, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **extras
    ) -> Dict[Text, torch.Tensor]:
        """
        tokens: [batch_size, seq_len],
        labels: [batch_size, seq_len],
        """
        
        token_vec = self.embedding(tokens)  # [batch_size, seq_len, embeddings]
        predict_logits = self.prediction_head(token_vec)['logits']
        
        # [batch_size, seq_len]
        mask = tokens[self._token_namespace]['mask']
        labels[~mask] = 0

        if labels is not None:
            if self.crf is not None:
                loss = -self.crf(
                    inputs=predict_logits,
                    tags=labels,
                    mask=mask
                )
            else:
                loss_func = torch.nn.CrossEntropyLoss()
                masked_labels = labels.masked_fill(
                    mask=~mask,
                    value=-100
                )
                loss = loss_func(predict_logits.flatten(0, 1), masked_labels.flatten())
        else:
            loss = None
                
        # also we need to do the logits decoding.
        # decode the result.
        if self.crf is not None:
            # decode with viterbi
            viterbi_decoding = self.crf.viterbi_tags(
                logits=predict_logits,
                mask=mask,
                top_k=None
            )
            predictions = [x[0] for x in viterbi_decoding]
            
            class_probabilities = predict_logits * 0.0
            for i, instance_tags in enumerate(predictions):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
                    
        else:
            class_probabilities = predict_logits
            predictions = torch.argmax(class_probabilities, dim=-1).cpu().detach().tolist()

        # calculate accuracy
        if labels is not None:
            for metric in self.metrics.values():
                metric(
                    predictions=class_probabilities.view(-1, class_probabilities.shape[-1]),
                    gold_labels=labels.flatten(),
                    mask=mask.flatten()
                )
                
        return {
            'logits': predict_logits,
            'loss': loss,
            'tags': predictions
        }

    def get_metrics(self, reset: bool = False) -> Dict[Text, float]:
        """
        """
        metrics = {
            f'{key_met}::{key}': val for key_met, val_met in self.metrics.items()
            for key, val in val_met.get_metric(reset).items()
        }

        return metrics
    
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        """
        
        def decode_tags(tags: List[int]):
            """
            """
            return [
                self.vocab.get_token_from_index(tags, namespace=self._label_namespace) for tag in tags
            ]
            
        output_dict['tags'] = [decode_tags(t) for t in output_dict['tags']]
        
        return output_dict

    def calculate_span_probabilities(
        self,
        tokens: Dict[Text, torch.Tensor],
        labels: torch.Tensor,
        masks: torch.Tensor = None,
        **extras
    ) -> torch.Tensor:
        """
        This function takes in a set of span labels and calculate
         the probability of a span by marginalizing other places
         
        tokens: [batch_size, seq_len]
        labels: [batch_size, (c_size), seq_len, num_tags]
        masks: [batch_size, c_size] bool
        
        masks can be none only if labels are two dimensional,
        in that case we will use the token masks.
        """
        
        token_vec = self.embedding(tokens)  # [batch_size, seq_len, embeddings]
        predict_logits = self.prediction_head(token_vec)['logits']

        # print(predict_logits.shape)
        # print(labels.shape)
        # print(masks.shape)

        if labels.ndim == 2:
            if masks is None:
                masks = tokens[self._token_namespace]['mask']
            
            # marginalize all span label distribution
            if self.crf is not None:
                logits = self.crf.marginalize(
                    step_logits=predict_logits,
                    step_masks=masks
                )  # [batch_size, sequence_length, num_labels]
            else:
                logits = predict_logits
                
            return {
                'logits': logits.flatten(0, 1)[masks.flatten()],
                'labels': labels.flatten()[masks.flatten()]
            }
            
        elif labels.ndim == 4:
            # here the input is a sequence of constraints as labels to be decoded.
            masks = masks.flatten()
            
            predict_logits = predict_logits.unsqueeze(1).repeat(1, labels.size(1), 1, 1)
            predict_logits = predict_logits.flatten(0, 1)[masks]
            token_mask = tokens[self._token_namespace]['mask'].unsqueeze(1).repeat(1, labels.size(1), 1).flatten(0, 1)[masks]

            labels = labels.flatten(0, 1)[masks]
            binary_labels = extras['binary_labels'].flatten()[masks]
            
            if self.crf is not None:
                
                log_likelihood: torch.Tensor = self.crf(
                    predict_logits,
                    labels,
                    mask=token_mask,
                    aggregate=False
                )  # [batch_size * num_constraints]
                
                logits = torch.stack([(1 - log_likelihood.exp()).log(), log_likelihood], dim=1)
                
            else:
                predicted_log_probs = torch.softmax(predict_logits, dim=-1) * labels
                log_likelihood = torch.sum(torch.sum(predicted_log_probs * labels, dim=-1).log() * token_mask, dim=-1)

                logits = torch.stack([(1 - log_likelihood.exp()).log(),  log_likelihood], dim=1)

            return {
                'logits': logits,
                'labels': binary_labels
            }
            