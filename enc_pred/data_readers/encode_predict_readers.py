"""Encode-predict readers are a set of readers that is
used to read datasets from datasets of the predictions
made with pre-trained encoders.
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
from functools import partial

from ..utils.commons import __VIRTUAL_ROOT__


class SpanReader(DatasetReader, abc.ABC):
    """
    """
    def __init__(self, is_prediction: Optional[bool] = False):
        super().__init__()
        self.is_prediction = is_prediction

    def _reindex_spans(
        self, spans: List[Tuple[int, int]],
        fields: Dict[Text, Field]
    ) -> List[Tuple[int, int]]:
        """Reindex the span indices to the wordpiece index.

        Notice that here we'll insert a [0, 0] which
        is definitely the virtual root of the dataset.
        """
        offsets = fields['raw_sentence']['offsets']
        # reindexed = [(0, 0)]
        reindexed = []

        for boundary in spans:
            if offsets[boundary[0]] is not None:
                start_idx = offsets[boundary[0]][0]
            else:
                start_idx = offsets[boundary[0] + 1][0]
            
            if offsets[boundary[1]] is not None:
                end_idx = offsets[boundary[1]][1]
            else:
                end_idx = offsets[boundary[1] - 1][1]

            # only preserve meaningful spans
            if end_idx >= self.max_length:
                continue

            assert end_idx >= start_idx, f'Negative span range detected.'
            reindexed.append(SpanField(
                span_start=start_idx, span_end=end_idx,
                sequence_field=fields['tokens']))

        return ListField(reindexed)

    
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


@DatasetReader.register('universal-dependency')
class UniversalDependencyReader(SpanReader):
    """This is the dataset reader for universal dependency
    dataset, based on the conllu reader provided in
    python.
    """
    def __init__(
        self,
        max_length: int,
        pretrained_model: Text,
        task: Text,
        is_prediction: Optional[bool] = False
    ):
        """
        """
        super().__init__(is_prediction)
        self.model_name = pretrained_model
        self.max_length = max_length
        self.task = task

        self.tokenizer = PretrainedTransformerTokenizer(
            model_name=self.model_name,
            max_length=self.max_length
        )

        self.indexer = PretrainedTransformerIndexer(
            model_name=self.model_name,
            max_length=self.max_length
        )

        # For some Asian languages the CLF rel will not appear in the
        # dataset, so it is possible that we exaclude the label prediction
        # for calibration purpose.
        self._additional_mask = {'clf'}

    def validate(self, item: Dict[Text, List[Any]]) -> bool:
        """Validate the parsed data from conllu dataset
        to see whether the final result parsed correctly.
        """
        length = None

        for val in item.values():
            if length is None:
                length = len(val)

            else:
                assert length == len(val), f'The parsed item does not have equal length: {item}'

        return True

    @overrides
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

                    if self.validate(item):
                        yield self.text_to_instance(**item)

    def text_to_instance(
        self,
        id: List[int],
        form: List[Text],
        lemma: List[Text],
        upos: List[Text],
        xpos: List[Text],
        feats: List[Text],
        head: List[int],
        deprel: List[Text],
        deps: List[Optional[Text]],
        misc: List[Optional[Text]]
    ) -> Instance:
        """Convert a listlized item to allennlp
        instance, where the dataset is easily
        loaded with built-in dataloader.
        """
        gen_unit_spans = lambda x: [(i, i) for i in range(x)]
        # modify deprel
        _remove_unk_labels = lambda x: x if x not in self._additional_mask else __VIRTUAL_ROOT__
        deprel = [_remove_unk_labels(relation.split(':')[0]) for relation in deprel]

        spans = gen_unit_spans(len(form))
        fields = self._index_sentence(form)
        spans = self._reindex_spans(spans, fields)

        head = [hd if hd is not None else -1 for hd in head]

        # for each span set its parent (since VIRTUAL_ROOT) is appended we do not
        # need to update the index to 0 based.

        parent_ids = TensorField(
            torch.tensor(([-1] + head)[:spans.sequence_length()], dtype=torch.int64),
            padding_value=-1, dtype=torch.int64
        )

        # extract relationship
        dr_partial = partial(LabelField, label_namespace='deprel_labels')
        pt_partial = partial(LabelField, label_namespace='labels')
        deprel_labels = ListField(list(map(dr_partial, ([__VIRTUAL_ROOT__] + deprel)[:spans.sequence_length()])))
        pos_tag_labels = ListField(list(map(pt_partial, ([__VIRTUAL_ROOT__] + upos)[:spans.sequence_length()])))


        # Only apply parent mask if the task is 'deprel'
        if self.task == 'deprel':
            fields['parent_ids'] = parent_ids

            # TODO: move this into model
            parent_mask = torch.logical_and(
                0 <= parent_ids.tensor,
                parent_ids.tensor < spans.sequence_length()
            )
            cls_mask = torch.tensor([False] + [dr != __VIRTUAL_ROOT__ for dr in deprel], dtype=torch.bool)
            fields['parent_mask'] = TensorField(
                torch.logical_and(cls_mask, parent_mask),
                dtype=torch.bool
            )
                
        fields['spans'] = spans

        # Only generate labels for training
        if not self.is_prediction:
            if self.task == 'deprel':
                fields['labels'] = deprel_labels
            elif self.task == 'pos_tags':
                fields['labels'] = pos_tag_labels
            else:
                raise NotImplementedError

        return Instance(fields)

    def _reindex_spans(
        self, spans: List[Tuple[int, int]],
        fields: Dict[Text, Field]
    ) -> List[Tuple[int, int]]:
        """Reindex the span indices to the wordpiece index.

        Notice that here we'll insert a [0, 0] which
        is definitely the virtual root of the dataset.
        """
        offsets = fields['raw_sentence']['offsets']
        reindexed = [SpanField(
            span_start=0,
            span_end=0,
            sequence_field=fields['tokens']
        )]

        for boundary in spans:
            if offsets[boundary[0]] is not None:
                start_idx = offsets[boundary[0]][0]
            else:
                start_idx = offsets[boundary[0] + 1][0]
            
            if offsets[boundary[1]] is not None:
                end_idx = offsets[boundary[1]][1]
            else:
                end_idx = offsets[boundary[1] - 1][1]

            # only preserve meaningful spans
            if end_idx >= self.max_length:
                continue

            assert end_idx >= start_idx, f'Negative span range detected.'
            reindexed.append(SpanField(
                span_start=start_idx, span_end=end_idx,
                sequence_field=fields['tokens']))

        return ListField(reindexed)


@DatasetReader.register('wikiann')
class WikiAnnReader(SpanReader):
    """WikiAnn Reader is used to read data for NER in various amount of
    languages.
    """
    def __init__(
        self,
        max_length: int,
        pretrained_model: Text,
        cache_dir: Text = 'data/wikiann/cache',
        is_prediction: Optional[bool] = False
    ):
        """
        """
        super().__init__(is_prediction)
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
        
        self.dataset_stem = 'wikiann'
        self.cache_dir = cache_dir

    @overrides
    def _read(self, file_path: Union[Text, List[Text]]) -> Iterable[Instance]:
        """Read the wiki_ann data for the items.

        file_path: Pseudo file_path that serves as
        a way to download WkiAnn Remote Dataset.

        format: lang/split
        """

        if isinstance(file_path, str):
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
                spans = [(i, i) for i in range(len(ner_tags))]

                data_obj = {
                    'tokens': item['tokens'],
                    'spans': spans,
                    'labels': ner_tags
                }

                if self.validate(data_obj):
                    yield self.text_to_instance(**data_obj)

    def validate(self, item: Dict[Text, Any]) -> bool:
        """Check whether spans and labels are the same length.
        """
        return len(item['labels']) == len(item['spans'])

    def text_to_instance(
        self, tokens: List[Text],
        spans: List[Tuple[int, int]],
        labels: List[Text]
    ) -> Instance:
        """
        """
        fields = self._index_sentence(tokens)
        spans = self._reindex_spans(spans, fields)

        partial_init = partial(
            LabelField,
            label_namespace='labels',
            skip_indexing=True
        )

        if not self.is_prediction:
            labels = ListField(list(map(partial_init, labels[:spans.sequence_length()])))
            fields['labels'] = labels
        fields['spans'] = spans

        return Instance(fields)