"""This sequence tagging loaders are used to compare performance between
different sequence taggin mdoule w/t the crf module.
"""
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

from ..utils.commons import __VIRTUAL_ROOT__


@DatasetReader.register('sequence-tagging')
class SequenceTaggingReader(DatasetReader):
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
                    yield self.text_to_instance(**json.loads(line))
    
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
        fields['labels'] = self._label_mapping(labels, alignments=fields['raw_sentence'])
        
        return Instance(fields)

    
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
        
        offsets = alignments['offsets']
        assert len(offsets) == len(labels), f'{labels} and doest not have the same length as {len(labels)} != {len(offsets)}'
        
        sequence = ['O']
        
        for lb, of in zip(labels, offsets):
            # checks whether the label is BIO
            if lb.startswith('B-'):
                sequence.extend([lb] + [lb.replace('B', 'I', 1)] * (of[1] - of[0]))
            else:
                sequence.extend([lb] * (of[1] - of[0] + 1))
                
        if truncate and len(sequence) > self.max_length - 1:
            sequence = sequence[:self.max_length - 1]
            
        sequence.append('O')
                
        # create labels from sequence
        sequence = ListField([LabelField(label=lb_str) for lb_str in sequence])
        
        return sequence
    
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
    
    
@DatasetReader.register("ner-sequence-tagging")
class NERTaggingReader(SequenceTaggingReader):
    """This is the specific dataset reader for NER dataset.

    Args:
        SequenceTaggingReader (_type_): _description_
    """
    def __init__(
        self,
        pretrained_model: Text,
        max_length: int,
        is_prediction: Optional[bool] = False,
        cache_dir: Text = 'data/wikiann/cache'):
        """
        """
        super().__init__(
            pretrained_model,
            max_length,
            is_prediction
        )
        
        self.cache_dir = cache_dir
        self.dataset_stem = 'wikiann'
        
    def _label_mapping(
        self,
        labels: List[Text],
        alignments: MetadataField,
        truncate: bool = True) -> ListField:
        """The difference with wikiann
        is that the dataset is already indexed.
        """
        offsets = alignments['offsets']
        assert len(offsets) == len(labels), f'{labels} and doest not have the same length as {len(labels)} != {len(offsets)}'
        
        sequence = [0]
        
        for lb, of in zip(labels, offsets):
            if lb % 2 == 1:
                sequence.extend([lb] + [lb + 1] * (of[1] - of[0]))
            else:
                sequence.extend([lb] * (of[1] - of[0] + 1))
        
        # if len(sequence) > 512:
        #     print(len(sequence))
        if truncate and len(sequence) > self.max_length - 1:
            sequence = sequence[:self.max_length - 1]

        sequence.append(0)
        
        # create labels from sequence
        sequence = ListField([LabelField(label=lb_str, skip_indexing=True) for lb_str in sequence])
        
        return sequence

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

                data_obj = {
                    'tokens': item['tokens'],
                    'labels': ner_tags
                }

                yield self.text_to_instance(**data_obj)
                
                
@DatasetReader.register('pos-tagging-reader')
class POSTaggingReader(SequenceTaggingReader):
    """POSTaggingReader reads data from the conllu
    datafile of the universal dependency parsing.
    """
    def __init__(
        self,
        max_length: int,
        pretrained_model: Text,
        is_prediction: Optional[bool] = False
    ):
        super().__init__(
            pretrained_model=pretrained_model,
            max_length=max_length,
            is_prediction=is_prediction
        )

    def _read(self, file_path: Union[Text, List[Text]]) -> Iterable[Instance]:
        """
        """

        if isinstance(file_path, str):
            file_path = [file_path]

        for fp in file_path:
            with open(fp, 'r', encoding='utf-8') as file_:
                for sentence in parse_incr(file_):
                    if not sentence:
                        continue
                    item = {
                        'id': [],
                        'form': [],
                        'lemma': [],
                        'upos': [],
                        'xpos': [],
                        'feats': [],
                        'head': [],
                        'deprel': [],
                        'deps': [],
                        'misc': []
                    }
                    for token in sentence:
                        # This skips items with span annotation
                        if not isinstance(token['id'], int):
                            continue
                        for key in item:
                            item[key].append(token[key])

                    yield self.text_to_instance(
                        tokens=item['form'],
                        labels=item['upos']
                    )