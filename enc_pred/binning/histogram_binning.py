"""Here we implement a version of shared CW
histogram binning over the shared probability scores.
"""
from typing import Text, List, Tuple, Dict
from regex import R
import torch
import numpy as np
import json
import os
from ..metrics.calibration_error import ExpectedCalibrationError


class HistogramBinning(torch.nn.Module):
    """
    """
    def __init__(
        self,
        num_bins: int,
        map_back_to_logit: bool = True,
        epsilon: float = 1e-12
    ):
        """
        """
        super().__init__()
        
        # setting requires_grad to allow gradient propagation
        self.scaling_parameter = torch.nn.Parameter(
            data=torch.ones((num_bins,), dtype=torch.float32),
            requires_grad=False
        )
        self._epsilon = torch.nn.Parameter(
            data=torch.tensor(epsilon, dtype=torch.float32),
            requires_grad=False
        )

        self._num_bins = num_bins
        self._step_size = 1 / num_bins
        
        self.metric = ExpectedCalibrationError(
            num_bins=num_bins
        )
        
        self._map_back_to_logit: bool = map_back_to_logit
        
    def fit(
        self,
        data_loader: torch.utils.data.DataLoader,
    ):
        """This function takes data from a torch dataloader
        and process them with the metrics function to generate bins
        and accuracies, and it then sets bin output to accuracies.
        """
        
        for batch in data_loader:
            self.metric(
                predictions=batch['logits'],
                gold_labels=batch['labels']
            )
            
        # and we derive our calibration binning from the ECE
        self.scaling_parameter.data = self.scaling_parameter.new_tensor(
            self.metric.accuracy_vec
        )
        
        self.metric.reset()
        
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *args, **kwargs
    ) -> Dict[Text, torch.Tensor]:
        """Return the calibrated probability with proper
        adjustments.
        
        x: [batch_size, num_classes]
        """
        
        inv_link = torch.nn.Softmax(dim=-1)
        x = inv_link(logits)
        x[x > 1. - self._epsilon] = 1. - self._epsilon
        x[x < self._epsilon] = self._epsilon
        
        bin_idx = torch.div(x, self._step_size, rounding_mode='floor').to(torch.int64)
        bin_idx[bin_idx > self._num_bins - 1] = self._num_bins - 1
        
        scaling_parameter = self.scaling_parameter.unsqueeze(0).expand(bin_idx.size(0), -1)
        
        # [batch_size, num_classes]
        aggregated_probs = torch.gather(
            scaling_parameter, dim=-1,
            index=bin_idx
        ) + self._epsilon
        
        predictions = torch.argmax(aggregated_probs, dim=-1)

        binary_labels = (predictions == labels).to(dtype=torch.int64)
        positive_probs = torch.max(aggregated_probs, dim=-1)[0] # [batch_size]
        binary_probs = torch.stack((1 - positive_probs, positive_probs), dim=-1)

        scaled_logits = torch.scatter(
            input=((1 - positive_probs.unsqueeze(-1)) / (logits.size(-1) - 1.)).repeat(1, logits.size(-1)),
            dim=-1,
            index=predictions.unsqueeze(-1),
            src=positive_probs.unsqueeze(-1)
        ).log()
        
        return {
            'logits': scaled_logits,
            'labels': labels
        }
        
    def save(self, serialization_dir: Text):
        """
        """
        torch.save(self.state_dict(), os.path.join(serialization_dir, 'best.th'))
        
        # also dump a set of configurations
        with open(os.path.join(serialization_dir, 'config.json'), 'w', encoding='utf-8') as file_:
            json.dump(
                {
                    'num_bins': self._num_bins,
                    'map_back_to_logit': self._map_back_to_logit,
                    'epsilon': self._epsilon.cpu().detach().item()
                }, file_
            )

    @classmethod
    def from_dir(cls, serialization_dir: Text) -> "HistogramBinning":
        """
        """
        with open(os.path.join(serialization_dir, 'config.json'), 'r', encoding='utf-8') as file_:
            configs = json.load(file_)
            
            model = cls(**configs)
            model.load_state_dict(torch.load(
                os.path.join(serialization_dir, 'best.th')
            ))
            
        return model