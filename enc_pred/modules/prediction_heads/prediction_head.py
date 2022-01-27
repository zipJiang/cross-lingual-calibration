from allennlp.common.registrable import Registrable
from allennlp.data.vocabulary import Vocabulary
import torch
import abc
from typing import Text, Dict, List, Union, Tuple


class PredictionHead(torch.nn.Module, Registrable, abc.ABC):
    """This is the abstract class of all prediction heads.
    """
    def __init__(self,
        vocabulary: Vocabulary,
        input_dim: int
    ):
        """vocabulary is used to infer num_labels in the space.
        """
        super().__init__()
        self.vocabulary = vocabulary
        self.input_dim = input_dim

    def forward(
        self,
        span_repr: torch.Tensor,
        span_mask: torch.Tensor,
        **extra,
    ) -> Dict[Text, torch.Tensor]:
        """This is the main function of the module,
        where we are transforming span representations
        into predicted class labels,

        span_repr: of shape [batch_size, num_spans, feature_dim]
        span_mask: same as above [batch_size, num_spans]
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        """
        return self.vocab.get_vocab_size(namespace='labels')