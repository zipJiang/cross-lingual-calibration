"""This is a test on subsampled dataset to makesure
that the dataloading isn't destroyed by the subsampling process.
"""
import unittest
from typing import Text, Dict, Tuple, List, Callable, Optional
from enc_pred.data_readers.encode_predict_readers import UniversalDependencyReader
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from tqdm import tqdm


class TestSubsampledDatasetSample(unittest.TestCase):
    def setUp(self):
        """
        """
        self.test_data_path = "/brtx/604-nvme2/zpjiang/encode_predict/data/udparse_train/subsampled_en_ewt-ud-2.9-train.conllu"
        self.data_reader = UniversalDependencyReader(
            max_length=256,
            pretrained_model='bert-base-uncased',
            task='deprel',
            is_prediction=False
        )

    def tset_data_reading(self):
        """
        """
        dataloader = MultiProcessDataLoader(
            reader=self.data_reader,
            data_path=self.test_data_path,
            batch_size=4,
            shuffle=True,
            cuda_device=None
        )

        for batch in tqdm(dataloader):
            pass