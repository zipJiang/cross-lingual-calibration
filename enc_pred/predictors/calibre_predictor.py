"""Predictor for generating calibration prediction.
"""
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.nn.util import move_to_device
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.instance import Instance
import torch
from typing import Tuple, Dict, List, Optional, Any, Union, Text
from overrides import overrides
from tqdm import tqdm


@Predictor.register('span-label-predictor')
class SpanLabelPredictor(Predictor):
    """
    """
    def predict(self, file_path: Text) -> List[JsonDict]:
        """This function reads a file_path with data_loader and
        dataset_reader to get predictions.
        """
        dataloader = MultiProcessDataLoader(
            reader=self._dataset_reader,
            data_path=file_path,
            batch_size=64,
            shuffle=False,
            cuda_device=self.cuda_device
        )

        dataloader.index_with(self._model.vocab)

        return_list = []

        with torch.no_grad():
            gen_span_mask = lambda x : x[..., 0] != -1

            for batch in tqdm(dataloader):
                move_to_device(batch, self.cuda_device)
                result_dict = self._model(**batch)

                logits = result_dict['logits']
                selection_logits = result_dict['selection_logits'] if 'selection_logits' in result_dict else None

                labels = batch['labels']
                selection_labels = None
                if selection_logits is not None:
                    selection_labels = batch['parent_ids']

                # compare logits
                span_mask = gen_span_mask(batch['spans'])

                if 'parent_mask' in batch:
                    num_spans = span_mask.sum(dim=-1).unsqueeze(1).extend(-1, span_mask.shape[1])
                    span_mask = torch.logical_and(span_mask, batch['parent_mask'])

                # return_dict['labels'].append(labels[span_mask])
                # return_dict['logits'].append(logits[span_mask])

                # if selection_logits is not None:
                #     return_dict['selection_logits'].append(selection_logits[span_mask])
                #     return_dict['selection_labels'].append(selection_labels[span_mask])

                if selection_logits is None:
                    for lgt, lb, num in zip(
                        logits[span_mask].cpu().split(1, dim=0),
                        labels[span_mask].cpu().split(1, dim=0)
                    ):
                        item_dict = {
                            'logit': lgt.tolist()[0],
                            'label': lb.item()
                        }
                        return_list.append(item_dict)
                else:
                    for lgt, lb, slgt, slb, num in zip(
                        logits[span_mask].cpu().split(1, dim=0),
                        labels[span_mask].cpu().split(1, dim=0),
                        selection_logits[span_mask].cpu().split(1, dim=0),
                        selection_labels[span_mask].cpu().split(1, dim=0),
                        num_spans[span_mask].cpu().split(1, dim=0)
                    ):
                        item_dict = {
                            'logit': lgt.tolist()[0],
                            'label': lb.item(),
                            'selection_logit': slgt.tolist()[0][:num],
                            'selection_label': slb.item()
                        }
                        return_list.append(item_dict)

        return return_list