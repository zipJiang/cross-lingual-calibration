from importlib.metadata import MetadataPathFinder
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Token
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.fields.field import Field
from allennlp.data.fields.text_field import TextField
from allennlp.data.fields.label_field import LabelField
from allennlp.data.fields.tensor_field import TensorField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.fields.span_field import SpanField
from allennlp.data.fields.list_field import ListField
from allennlp.data.instance import Instance
import torch
from typing import Text, List, Union, Dict, Tuple, Optional, Any, Iterable
from conllu import parse, parse_incr
from overrides import overrides
import datasets
import abc
import json
from functools import partial


class SentPredReader(DatasetReader):
    def __init__(self, is_prediction: Optional[bool] = False):
        """
        """
        super().__init__()
        self.is_prediction = is_prediction


@DatasetReader.register("xnli")
class XNLIReader(SentPredReader):
    def __init__(
        self,
        pretrained_model: Text,
        max_length: int,
        is_prediction: Optional[bool] = False,
    ):
        """
        """
        super().__init__(is_prediction)
        self.model_name = pretrained_model
        self.max_length = max_length

        self.tokenizer = PretrainedTransformerTokenizer(
            model_name=self.model_name,
            max_length=self.max_length,
            # set add_special_tokens to False to allow concat
            add_special_tokens=False
        )

        self.indexer = PretrainedTransformerIndexer(
            model_name=self.model_name,
            max_length=2 * self.max_length
        )

        self.dataset_stem = 'xnli'

    @overrides
    def _read(self, file_path: Union[Text, List[Text]]) -> Iterable[Instance]:
        """The huggingface curatede split is limited
        in language. We use the raw XNLI file.
        """
        if isinstance(file_path, str):
            file_path = [file_path]

        for fp in file_path:
            # get maximum reading config
            fp_split = fp.split('[')
            fp = fp_split[0]
            num_items = fp_split[1] if len(fp_split) > 1 else None
            num_items = int(num_items[1:-1]) if num_items is not None else float("Inf")
            with open(fp, 'r', encoding='utf-8') as file_:
                for lidx, line in enumerate(file_):
                    if lidx >= num_items:
                        break
                    item = json.loads(line)
                    gold_label = item['gold_label']
                    premise, hypothesis = item['sentence1'], item['sentence2']

                    yield self.text_to_instance(
                        premise=premise,
                        hypothesis=hypothesis,
                        gold_label=gold_label
                    )

    def text_to_instance(
        self,
        premise: Text,
        hypothesis: Text,
        gold_label: Text
    ) -> Instance:
        """Generate instance that has one sentence-pair
        embedding, and gold label display.
        """
        field_dict = self.tokenize(
            premise, hypothesis
        )

        field_dict['labels'] = LabelField(
            label=gold_label,
            label_namespace='labels',
            skip_indexing=False
        )

        return Instance(fields=field_dict)

    def tokenize(self, premise: Text, hypothesis: Text) -> Dict[Text, Field]:
        """
        """
        tok_p = self.tokenizer.tokenize(premise)
        tok_h = self.tokenizer.tokenize(hypothesis)

        tokens = self.tokenizer.add_special_tokens(tokens1=tok_p, tokens2=tok_h)

        return_dict = {
            'tokens': TextField(
                tokens=tokens,
                token_indexers={
                    'pieces': self.indexer
                }
            ),
            'meta-data': MetadataField(
                {
                    'premise': premise,
                    'hypothesis': hypothesis
                }
            )
        }

        return return_dict