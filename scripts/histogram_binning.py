"""Run the histogram binning training and evaluation
"""
import argparse
import os
import json
import torch
from typing import Text, Tuple, List, Dict, Optional, Union, Any
from enc_pred.binning.histogram_binning import HistogramBinning
from enc_pred.metrics import ExpectedCalibrationError, BrierScore
from allennlp.training.metrics.metric import Metric
from tqdm import tqdm


def parse_args():
    """
    """
    
    parser = argparse.ArgumentParser(
        """Running histogram binning (one-vs-all)
        with combined binning scheme.
        """
    )
    parser.add_argument(
        '--train_path', action='store', dest='train_path',
        type=str, required=True, help='Training data path.'
    )

    parser.add_argument(
        '--eval_path', action='store', dest='eval_paths',
        type=str, required=True, help='Evaluate the result from the following paths.',
        nargs='+'
    )
    
    parser.add_argument(
        '--serialization_dir', action='store', dest='serialization_dir',
        type=str, required=True, help='Where to write the result data.'
    )

    parser.add_argument(
        '--num_bins', action='store', dest='num_bins',
        type=int, required=False, default=100, help='Num bins.'
    )

    parser.add_argument(
        '--logit_key', action='store', dest='logit_key',
        type=str, required=False, help='logit_key to load logit',
        default='logit'
    )
    
    parser.add_argument(
        '--label_key', action='store', dest='label_key',
        type=str, required=False, help='label_key to load labels',
        default='label'
    )
    
    return parser.parse_args()


def collate_fn_2d(
    batch: List[Dict[Text, torch.Tensor]]
) -> Dict[Text, torch.Tensor]:
    """Here we assume that the input is 1d
    """

    aggs = [item['logits'] for item in batch]
    if aggs[0].ndim > 0:
        max_length = max([a.size(-1) for a in aggs])
        aggs = [torch.nn.functional.pad(a, (0, max_length - a.size(-1)), mode='constant', value=-2e8) for a in aggs]

    return {
        'logits': torch.stack(aggs, dim=0),
        'labels': torch.stack([item['labels'] for item in batch], dim=0)
    }
            

class CalibrationDataset(torch.utils.data.Dataset):
    """Dataset only used for calibration with non-parametric methods
    """
    def __init__(
        self,
        file_path: Text,
        logit_key: Optional[Text] = 'logit',
        label_key: Optional[Text] = 'label'
    ):
        """
        """
        super().__init__()
        self._data = []
        self._logit_key = logit_key
        self._label_key = label_key

        with open(file_path, 'r', encoding='utf-8') as file_:
            for line in file_:
                self._data.append(json.loads(line))
                
    def __getitem__(self, x: int) -> Dict[Text, torch.Tensor]:
        """
        """
        return {
            'logits': torch.tensor(self._data[x][self._logit_key], dtype=torch.float32),
            'labels': torch.tensor(self._data[x][self._label_key], dtype=torch.int64),
            'num_labels': torch.tensor(len(self._data[x][self._logit_key]), dtype=torch.int64)
        }
        
    def __len__(self) -> int:
        """
        """
        return len(self._data)

        
class NonGradientTrainer:
    """
    """
    def __init__(
        self,
        model: HistogramBinning,
        serialization_dir: Text,
        # because it is non-parametric it will be without epochs
        train_data_loader: Optional[torch.utils.data.DataLoader] = None,
        eval_data_loaders: Optional[Dict[Text, torch.utils.data.DataLoader]] = None,
        evaluation_metrics: Optional[Dict[Text, Metric]] = None,
    ):
        self._model = model
        self._model.train()
        self._serialization_dir = serialization_dir
        
        self._metrics = evaluation_metrics
        self._train_data_loader = train_data_loader
        self._eval_data_loaders = eval_data_loaders
        
    def _save_model(self):
        """
        """
        assert not os.path.isdir(self._serialization_dir), f"{self._serialization_dir} already existed!"
        os.makedirs(name=self._serialization_dir, exist_ok=False)
        
        # save a configuration
        self._model.save(self._serialization_dir)
        
    @classmethod
    def from_dir(cls, serialization_dir: Text) -> "NonGradientTrainer":
        """This function loads a blank trainer from
        the serialization dir.
        """
        
        model = HistogramBinning.from_dir(
            serialization_dir
        )
        
        return cls(
            serialization_dir=serialization_dir,
            model=model
        )
        
    def train(self) -> Union[None, Dict[Text, Any]]:
        
        assert self._train_data_loader is not None, "train_data_loader is not set, need to call set_train_data_loader first."
        self._model.fit(self._train_data_loader)

        # TODO: adding a model save function
        self._save_model()
        
        
        if self._eval_data_loaders is not None and self._metrics is not None:
            result_dicts = {}
            # return self.evaluate(self._eval_data_loader, self._metrics)
            for key, eval_loader in self._eval_data_loaders.items():
                result_dicts[key] = self.evaluate(eval_loader, self._metrics)
                
            return result_dicts
        
        return
            
    def evaluate(self, data_loader: torch.utils.data.DataLoader, metrics: Dict[Text, Metric]) -> Dict[Text, Any]:
        """Evaluating the model against provided dataloader
        and metrics.
        """
        output_dict = {}

        # calculate original metrics
        for batch in data_loader:
            for metric_name, metric in metrics.items():
                metric(
                    predictions=batch['logits'],
                    gold_labels=batch['labels']
                .detach())
        # combining all metrics output
        for metric_name, metric in metrics.items():
            metric_dict = metric.get_metric(reset=True)
            for key, val in metric_dict.items():
                output_dict[f"ori::{metric_name}::{key}"] = val

        # will process the data for two rounds, with metrics reset in the middle
        for batch in data_loader:
            return_val = self._model(**batch)

            for metric_name, metric in metrics.items():
                metric(
                    predictions=return_val['logits'],
                    gold_labels=return_val['labels']
                )
                
        # combining all metrics output
        for metric_name, metric in metrics.items():
            metric_dict = metric.get_metric(reset=True)
            for key, val in metric_dict.items():
                output_dict[f"scaled::{metric_name}::{key}"] = val

        return output_dict
        
        
def main():
    """
    """
    args = parse_args()
    
    model = HistogramBinning(
        num_bins=args.num_bins,
        map_back_to_logit=True
    )

    train_dataset = CalibrationDataset(
        file_path=args.train_path,
        logit_key=args.logit_key,
        label_key=args.label_key
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=False,
        collate_fn=collate_fn_2d
    )

    eval_datasets = [
        CalibrationDataset(
            file_path=eval_path,
            logit_key=args.logit_key,
            label_key=args.label_key
        )
        for eval_path in args.eval_paths]
    eval_dataloaders = {os.path.basename(filename)[:-1]: torch.utils.data.DataLoader(
        edst, batch_size=1024, shuffle=False,
        collate_fn=collate_fn_2d
    ) for filename, edst in zip(args.eval_paths, eval_datasets)}

    metrics_dict = {
        # 'brier-score': BrierScore(),
        'ece': ExpectedCalibrationError(
            num_bins=100, steps=[2, 4, 5, 10, 20]
        )
    }
    
    trainer = NonGradientTrainer(
        model=model,
        serialization_dir=args.serialization_dir,
        train_data_loader=train_dataloader,
        eval_data_loaders=eval_dataloaders,
        evaluation_metrics=metrics_dict
    )
    
    evaluation_dicts = trainer.train()
    print(evaluation_dicts)
    
    # also try to write the evaluation result to the serialization dir
    if evaluation_dicts is not None:
        os.makedirs(os.path.join(args.serialization_dir, "eval"), exist_ok=True)
        
        for filename, eval_dict in evaluation_dicts.items():
            with open(os.path.join(args.serialization_dir, "eval", filename), 'w', encoding='utf-8') as file_:
                json.dump(eval_dict, file_, indent=4)
    
    
if __name__ == '__main__':
    main()