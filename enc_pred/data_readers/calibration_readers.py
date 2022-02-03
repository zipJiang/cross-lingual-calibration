"""Calibration Readers will read logits prediction and label files
and try to generate trainable batches for calibration training.
"""
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields.field import Field
from allennlp.data.fields.tensor_field import TensorField
from allennlp.data.fields.label_field import LabelField

from typing import Text, Dict, List, Tuple, Optional, Any, Union, Iterable
import torch
import json


@DatasetReader.register('calibration-reader')
class CalibrationReader(DatasetReader):
    def __init__(
        self,
        is_prediction: bool = False,
        logits_key: Text = 'logit',
        labels_key: Text = 'label'
    ):
        super().__init__()
        self.is_prediction = is_prediction
        self._logits_key = logits_key
        self._labels_key = labels_key

    def _read(self, file_path: Union[Text, List[Text]]) -> Iterable[Instance]:
        """
        """
        with open(file_path, 'r', encoding='utf-8') as file_:

            for line in file_:
                data = json.loads(line)

                logit = data[self._logits_key]
                label = data[self._labels_key]

                yield self.text_to_instance(logit, label)
    
    def text_to_instance(self, logit: List[float], label: int) -> Instance:
        """
        """
        logit_field = TensorField(
            torch.tensor(logit, dtype=torch.float32),
            padding_value=-2e8,
            dtype=torch.float32
        )
        label_field = LabelField(
            label,
            skip_indexing=True
        )

        num_labels_field = TensorField(
            torch.tensor(len(logit), dtype=torch.int64),
            padding_value=-1,
            dtype=torch.float32
        )

        return_dict = {
            'logits': logit_field,
            'num_labels': num_labels_field
        }

        if not self.is_prediction:
            return_dict['labels'] = label_field

        return Instance(return_dict)