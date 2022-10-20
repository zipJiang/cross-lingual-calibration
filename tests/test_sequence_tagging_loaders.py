"""Testing the sampled sequence tagging
dataloader for POS and NER.
"""
import unittest
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from typing import Text, Dict, Tuple, List, Callable, Optional
from tqdm import tqdm
from enc_pred.data_readers import POSTaggingReader, NERTaggingReader


class TestSequenceTaggingReader(unittest.TestCase):
    def setUp(
        self,
    ):
        self._pos_tagging_path = '/brtx/604-nvme2/zpjiang/encode_predict/data/udparse_train/subsampled_datasets/subsubsampled_en_ewt-ud-2.9-train.conllu'
        self._ner_tagging_path = 'en/train[:100]'
        
        self._pos_tagging_reader = POSTaggingReader(
            max_length=256,
            pretrained_model='bert-base-uncased',
            is_prediction=False
        )

        self.ner_tagging_reader = NERTaggingReader(
            max_length=256,
            pretrained_model='bert-base-uncased',
            is_prediction=False
        )
        
    def test_pos_tagging_path(self):
        """
        """

        vocabulary = Vocabulary.from_instances(
            self._pos_tagging_reader.read(self._pos_tagging_path)
        )
        
        dataloader = MultiProcessDataLoader(
            reader=self._pos_tagging_reader,
            data_path=self._pos_tagging_path,
            batch_size=4,
            shuffle=False,
            cuda_device=None,
        )
        
        dataloader.index_with(vocabulary)

        for batch in tqdm(dataloader):
            print(batch)