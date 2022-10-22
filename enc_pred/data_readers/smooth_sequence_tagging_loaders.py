"""This sequence tagging loaders are used to marginalize all spans
in a sentence that could be marginalized.
"""
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

from ..utils.commons import __VIRTUAL_ROOT__


class SmoothSequenceTaggingReader(DatasetReader, abc.ABC):
    """This reader should be used for two different
     tasks: 1 POS 2 NER extraction.
    """
    def __init__(
        self,
        pretrained_model: Text,
        max_length: int,
        is_prediction: Optional[bool] = False):
        super().__init__()
        self.is_prediction = is_prediction
        self.model_name = pretrained_model
        self.max_length = max_length

        self.tokenizer = PretrainedTransformerTokenizer(
            model_name=self.model_name,
            max_length=self.max_length
        )

        self.indexer = PretrainedTransformerIndexer(
            model_name=self.model_name,
            max_length=self.max_length
        )
        
    def _read(self, file_path: Union[Text, List[Text]]) -> Iterable[Instance]:
        """Notice that this reader reads the result of
        dataset bootstrapping, ideally this will be a file of jsonl
        """
        
        if not isinstance(file_path, list):
            file_path = [file_path]

        for fp in file_path:
            with open(fp, 'r', encoding='utf-8') as file_:
                for line in file_:
                    instance = self.text_to_instance(**json.loads(line))
                    if instance is not None:
                        yield instance
    
    def text_to_instance(
        self,
        tokens: List[Text],
        labels: List[Text],
        **extras
    ) -> Instance:
        """Process one input instance, that should
        go for the DatasetReader. Preferrably this
        method should not be overwritten.
        """
        
        fields = self._index_sentence(sentence=tokens)

        mapped_labels = self._label_mapping(labels, alignments=fields['raw_sentence'])
        if mapped_labels is None:
            return None
            
        # fields['labels'] = mapped_labels['constraints']
        # fields['binary_labels'] = mapped_labels['binary_labels']
        fields.update(mapped_labels)
        
        return Instance(fields)

    
    @abc.abstractmethod
    def _label_mapping(
        self,
        labels: List[Text],
        alignments: MetadataField,
        truncate: bool = True,
    ) -> ListField:
        """This function takes a list of label str, 
        and try to combine the labels against the sentence-level
        word to token aligmeents.
        """
        raise NotImplementedError
    
    def _index_sentence(self, sentence: List[Text]) -> Dict[Text, Field]:
        """Generating part of the fields w.r.t. sentence processing.
        """
        pieces, offsets = self.retokenize(sentence, truncate=True)
        fields = {
            'tokens': TextField(list(map(Token, pieces)), {'pieces': self.indexer}),
            'raw_sentence': MetadataField(
                {
                    'sentence': sentence,
                    'pieces': pieces,
                    'offsets': offsets
                }
            )
        }

        return fields
    
    def retokenize(
        self, sentence: List[Text],
        truncate: bool = True
    ) -> Tuple[List[Text], List[Optional[Tuple[int, int]]]]:
        """
        """
        pieces, offsets = self.tokenizer.intra_word_tokenize(sentence)
        pieces = list(map(str, pieces))
        if truncate and len(pieces) > self.max_length:
            pieces = pieces[:self.max_length][:-1] + [pieces[-1]]

        return pieces, offsets
    
    
@DatasetReader.register("smooth-ner")
class SmoothNERTaggingReader(SmoothSequenceTaggingReader):
    """This is the specific dataset reader for NER dataset.

    Args:
        SequenceTaggingReader (_type_): _description_
    """
    def __init__(
        self,
        pretrained_model: Text,
        max_length: int,
        max_span_length: int,
        negative_subsample_probability: float = .01,
        is_prediction: Optional[bool] = False,
        cache_dir: Text = 'data/wikiann/cache'):
        """
        """
        super().__init__(
            pretrained_model,
            max_length,
            is_prediction
        )
        
        self.max_span_length = max_span_length
        self.negative_subsample_probability = negative_subsample_probability
        self.cache_dir = cache_dir
        self.dataset_stem = 'wikiann'
        
    def _label_mapping(
        self,
        labels: List[int],
        alignments: MetadataField,
        truncate: bool = True) -> Union[Dict[Text, Field], None]:
        """The difference with wikiann
        is that the dataset is already indexed.
        """
        offsets = alignments['offsets']
        assert len(offsets) == len(labels), f'{labels} and doest not have the same length as {len(labels)} != {len(offsets)}'

        span_dict = self._extract_spans(labels, offsets)
        
        if not span_dict:
            return None

        # generate a set of labels
        # unfortunately 7 is a magic number
        num_terms = min(self.max_span_length, len(alignments['pieces']))
        num_constraints = 3 * (2 * len(alignments['pieces']) - num_terms + 1) * num_terms // 2
        base = torch.ones(num_constraints, len(alignments['pieces']), 7)

        # try to fill the base
        binary_labels = []
        current_constraint_id = 0
        for i in range(1, base.size(1) - 1, 1):
            for j in range(i, min(base.size(1) - 1, i + self.max_span_length), 1):
                for l in range(1, 6, 2):
                    base[current_constraint_id, i:j + 1] = 0
                    base[current_constraint_id, i, l] = 1
                    if j > i:
                        base[current_constraint_id, i + 1:j + 1, l + 1] = 1
                    if j + 1 < base.size(1):
                        base[current_constraint_id, j + 1, l + 1] = 0
                        
                    current_constraint_id += 1
                    # check whether this label is correct

                    if i in span_dict and j in span_dict[i] and span_dict[i][j] == l:
                        binary_labels.append(1)
                    else:
                        binary_labels.append(0)

        # we should not subsample the dataset because it will change the base-rate
        binary_labels = torch.tensor(binary_labels + [0] * (num_constraints - len(binary_labels)))
        masks = torch.tensor([1] * current_constraint_id + [0] * (num_constraints - current_constraint_id), dtype=torch.bool)
        
        random_mask = torch.bernoulli(
            torch.full(size=(num_constraints,), fill_value=self.negative_subsample_probability, dtype=torch.float32)
        ) > .5
        
        random_mask = torch.logical_or(random_mask, binary_labels.bool())
        masks = torch.logical_and(random_mask, masks)

        return {
            'masks': TensorField(masks, padding_value=0, dtype=torch.bool),
            'labels': TensorField(base, padding_value=1, dtype=torch.float32),
            'binary_labels': TensorField(binary_labels, padding_value=0, dtype=torch.int32)
        }

    def _extract_spans(self,
                       labels: List[int], offsets: List[Tuple[int, int]]) -> Union[Dict[int, Dict[int, int]], None]:
        """return a tree-like structure to determine correct label for the current
        span under construction.
        """
        word_spans = []
        starts = None
        ends = None
        clabel = None
        for lidx, l in enumerate(labels):
            if l == 0 or l % 2 == 1:
                if clabel is not None:
                    ends = lidx - 1
                    word_spans.append((starts, ends, clabel))

                if l == 0:
                    starts = None
                    clabel = None
                else:
                    starts = lidx
                    clabel = l
                    
        if clabel is not None:
            word_spans.append((starts, len(labels) - 1, clabel))
            
        # covert spans to offsets
        span_dict = {}
        
        for span in word_spans:
            starts = offsets[span[0]][0]
            ends = offsets[span[1]][-1]
            
            if ends >= self.max_length - 1:
                continue
            
            if starts not in span_dict:
                span_dict[starts] = {}
            
            span_dict[starts][offsets[span[1]][-1]] = span[2]

        return span_dict

    def _read(
        self,
        file_path: Union[Text, List[Text]]
    ) -> Iterable[Instance]:
        """
        """
        
        if not isinstance(file_path, list):
            file_path = [file_path]

        for fp in file_path:
            lang, split = fp.split('/')
            
            dataset = datasets.load_dataset(
                path=self.dataset_stem,
                name=lang,
                split=split,
                cache_dir=self.cache_dir
            )

            # Here the dataset should be of BIO format
            for item in dataset:
                ner_tags = item['ner_tags']

                # notice that here if the sentence is longer than max_length,
                # it is already discarded
                if len(ner_tags) > self.max_length:
                    continue

                data_obj = {
                    'tokens': item['tokens'],
                    'labels': ner_tags
                }

                instance = self.text_to_instance(**data_obj)
                
                if instance is not None:
                    yield instance
                