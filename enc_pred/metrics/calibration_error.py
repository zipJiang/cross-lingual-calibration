from allennlp.training.metrics import Metric
import torch
from overrides import overrides
from typing import Text, Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


@Metric.register('expected-calibration-error')
class ExpectedCalibrationError(Metric):
    """Computes labeled expected calibration error.
    """
    def __init__(
        self,
        num_bins: int = 10,
        steps: Optional[List[int]] = None,
        epsilon: float = 1e-12):
        """
        """
        super().__init__()
        self._num_bins = num_bins
        self._step_size = 1 / self._num_bins
        self._grouping_steps = steps
        if self._grouping_steps is not None:
            self._validate_grouping_steps()
        self._counter = np.zeros(self._num_bins, dtype=np.float32)
        self._confidence_list = np.zeros(self._num_bins, dtype=np.float32)
        self._accuracy_indicators = np.zeros(self._num_bins, dtype=np.float32)
        self._epsilon = epsilon

    def _validate_grouping_steps(self):
        """
        """
        for step_size in self._grouping_steps:
            assert self._num_bins % step_size == 0, f"sp: {step_size} is not compatible with #{self._num_bins}!"

    @property
    def num_bins(self) -> int:
        """
        """
        return self._num_bins

    @num_bins.setter
    def num_bins(self, num_bins: int):
        """
        """
        self._num_bins = num_bins
        self._step_size = 1 / self._num_bins
        self.reset()

    @overrides
    def __call__(self, predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """When called, aggregate scores according to the
        way we needed to calculate the get_metric func.

        predictions: --- [batch_size, num_labels]
        gold_labels: --- [batch_size]
        mask: --- [batch_size]
        """

        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        if mask is not None:
            predictions = predictions[mask]
            gold_labels = gold_labels[mask]

        confidence = torch.nn.functional.softmax(predictions, dim=-1).cpu().numpy()

        label = gold_labels.cpu().numpy()
        predicted_confidence, pred = confidence.max(axis=1), confidence.argmax(axis=1)

        is_correct = pred == label
        is_correct = is_correct.astype(np.float32)
        predicted_confidence = np.expand_dims(predicted_confidence, axis=1)
        is_correct = np.expand_dims(is_correct, axis=1)
        bin_indices = (predicted_confidence // self._step_size).astype(np.int64)
        # upper bounding the overflow values
        bin_indices[bin_indices >= self._num_bins] = self._num_bins - 1
        indicator_adding_matrix = np.zeros((is_correct.shape[0], self._num_bins), dtype=np.float32)
        confidence_adding_matrix = np.zeros((is_correct.shape[0], self._num_bins), dtype=np.float32)
        counter_adding_matrix = np.zeros((is_correct.shape[0], self._num_bins), dtype=np.float32)

        np.put_along_axis(indicator_adding_matrix, bin_indices,
                          is_correct.astype(np.float32), axis=1)
        np.put_along_axis(confidence_adding_matrix, bin_indices,
                          predicted_confidence, axis=1)
        np.put_along_axis(counter_adding_matrix, bin_indices,
                          1, axis=1)

        self._accuracy_indicators += indicator_adding_matrix.sum(axis=0)
        self._confidence_list += confidence_adding_matrix.sum(axis=0)
        self._counter += counter_adding_matrix.sum(axis=0)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[Text, Any]:
        """Notice that ECE could be calculated directly from 1/n * |p - I[y = y^]|,
        here we maintain the counter and the categorical information for plotting
        and analysis.
        """
        counter_sum = self._counter.sum()
        category_ce = np.absolute(self._accuracy_indicators - self._confidence_list).sum() / counter_sum
        return_dict = {
            'ECE': category_ce.item()
        }

        # add combined calibration result
        for step_size in self._grouping_steps:
            accuracy_indicator = self._accuracy_indicators.reshape(-1, step_size).sum(axis=-1)
            confidence_list = self._confidence_list.reshape(-1, step_size).sum(axis=-1)

            ece_at_step_size = np.absolute(accuracy_indicator - confidence_list).sum() / counter_sum
            return_dict[f'ECE-#{self._num_bins // step_size}'] = ece_at_step_size.item()

        if reset:
            self.reset()

        return return_dict

    @overrides
    def reset(self):
        self._counter = np.zeros(self._num_bins, dtype=np.float32)
        self._confidence_list = np.zeros(self._num_bins, dtype=np.float32)
        self._accuracy_indicators = np.zeros(self._num_bins, dtype=np.float32)

    def plot_confidence_hist(self) -> matplotlib.figure.Figure:
        """This function will generate a confidence histogram
        for the self._counter.
        """
        x = np.arange(0, 1 - self._epsilon, self._step_size)
        assert x.shape[0] == self._num_bins, f'the constructed shape of range {x.shape[0]} does not equal the number of number of bins #{self._num_bins}'

        y = self._counter.copy() / self._counter.sum()

        fig, axes = plt.subplots(1, 1)
        axes.set_title('Confidence Histogram')
        axes.bar(x, y, width=self._step_size, align='edge')
        axes.set_xlabel('confidence')
        axes.set_ylabel('dist')

        return fig

    def plot_reliability_diag(self) -> matplotlib.figure.Figure:
        """This function will generate a reliability
        disagram for the ECE.
        """

        x = np.arange(0, 1 - self._epsilon, self._step_size)
        assert x.shape[0] == self._num_bins, f'the constructed shape of range {x.shape[0]} does not equal the number of number of bins #{self._num_bins}'

        y_accu = self._accuracy_indicators / (self._counter + self._epsilon)

        fig, axes = plt.subplots(1, 1)
        axes.set_title("Reliability Diagram")

        axes.bar(x, y_accu, width=self._step_size, align='edge')
        axes.plot([0, 1], [0, 1], color='red')

        axes.set_xlabel('confidence')
        axes.set_ylabel('accuracy')

        return fig


# TODO: add a lazy constructor that construct metric from label
# or need to calculate num_classes based on input data.

# TODO: make mask truly available
# TODO: check whether we need to apply softmax cross-entropy examination
@Metric.register('class-calibration-error')
class ClassCalibrationError(Metric):
    """Compute class calibration error disregard
    of correct class label.
    """
    def __init__(self, num_bins: int, num_labels: int, epsilon: float = 1e-12):
        super().__init__()
        self._num_bins = num_bins
        self._num_labels = num_labels
        self._step_size = 1 / self._num_bins
        self._counter = np.zeros((self._num_bins, self._num_labels), dtype=np.float32)
        self._confidence_list = np.zeros((self._num_bins, self._num_labels), dtype=np.float32)
        self._label_dist = np.zeros((self._num_bins, self._num_labels), dtype=np.float32)
        self._neg_label_dist = np.zeros((self._num_bins, self._num_labels), dtype=np.float32)
        self._epsilon = epsilon

    @overrides
    def reset(self):
        self._counter = np.zeros((self._num_bins, self._num_labels), dtype=np.float32)
        self._confidence_list = np.zeros((self._num_bins, self._num_labels), dtype=np.float32)
        self._label_dist = np.zeros((self._num_bins, self._num_labels), dtype=np.float32)
        self._neg_label_dist = np.zeros((self._num_bins, self._num_labels), dtype=np.float32)

    @property
    def num_bins(self) -> int:
        """
        """
        return self._num_bins

    @num_bins.setter
    def num_bins(self, num_bins: int):
        """
        """
        self._num_bins = num_bins
        self._step_size = 1 / self._num_bins
        self.reset()

    @overrides
    def __call__(self, predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor]):
        """When called, aggregate scores according to each bin for
        each label.

        predictions: --- [batch_size, num_labels]
        gold_labels: --- [batch_size]
        mask: --- [batch_size]
        """
        predictions, gold_labels = self.detach_tensors(predictions, gold_labels, mask)

        confidence = predictions.cpu().numpy()
        label = gold_labels.cpu().numpy()

        # [batch_size, num_bins, num_labels]
        expanded_confidence = np.tile(np.expand_dims(confidence, axis=1), (1, self._num_bins, 1))
        # mask_out confidence if it is out of the range

        # [1, num_bins + 1, 1]
        division = np.expand_dims(np.arange(1.0 + self._epsilon, step=self._step_size), axis=(0, -1))

        confidence_mask = ((expanded_confidence >= division[:, :-1]) & (expanded_confidence < division[:, 1:])).astype(np.float32)
        
        self._confidence_list += np.sum(expanded_confidence * confidence_mask, axis=0)
        self._counter += np.sum(confidence_mask, axis=0)

        label_one_hot = np.zeros((label.shape[0], self._num_labels), dtype=np.float32)
        label_one_hot[np.arange(label.shape[0]), label] = 1.

        neg_label_one_hot = 1. - label_one_hot

        self._label_dist += np.sum(np.expand_dims(label_one_hot, axis=1) * confidence_mask, axis=0)
        self._neg_label_dist += np.sum(np.expand_dims(neg_label_one_hot, axis=1) * confidence_mask, axis=0)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[Text, Any]:
        """
        """

        classwise_ece = np.absolute(self._confidence_list - self._label_dist).sum() / (self._counter.sum() + self._epsilon)

        if reset:
            self.reset()

        return {
            'classwise-ECE': classwise_ece.item()
        }

    def plot_class_reliability_diag(self, class_idx: int) -> matplotlib.figure.Figure:
        """This function takes an additional specifier
        called class_idx, where we can choose a class
        idx to calculate the reliability diagram from.
        """
        x = np.arange(0, 1 - self._epsilon, self._step_size)
        assert x.shape[0] == self._num_bins, f'the constructed shape of range {x.shape[0]} does not equal the number of number of bins #{self._num_bins}'

        y_accu = self._label_dist[:, class_idx] / ((self._label_dist + self._neg_label_dist)[:, class_idx] + self._epsilon)

        fig, axes = plt.subplots(1, 1)
        axes.set_title("Reliability Diagram")

        axes.bar(x, y_accu, width=self._step_size, align='edge')
        axes.plot([0, 1], [0, 1], color='red')

        axes.set_xlabel('confidence')
        axes.set_ylabel('accuracy')

        return fig


@Metric.register('brier-score')
class BrierScore(Metric):
    def __init__(self):
        super().__init__()

        self._item_count = 0
        # we try to rely more on torch instead of numpy
        self._summation = None

    @overrides
    def __call__(
        self, predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        num_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ):
        """Shapes:
        predictions: [batch_size, num_labels]
        num_labels: [batch_size]
        gold_labels: [batch_size]
        mask: [batch_size]
        """
        predictions, gold_labels, num_labels, mask = self.detach_tensors(predictions, gold_labels, num_labels, mask)

        with torch.no_grad():
            if mask is not None:
                predictions = predictions[mask]
                gold_labels = gold_labels[mask]
            
            batch_size, label_dim = predictions.size()
            confidence = torch.nn.functional.softmax(predictions, dim=-1)

            targ = torch.nn.functional.one_hot(
                gold_labels,
                num_classes=label_dim
            )

            if num_labels is not None:
                element_mse = torch.square(confidence - targ).sum(dim=-1) / num_labels
            else:
                element_mse = torch.square(confidence - targ).mean(dim=-1)

            self._item_count += element_mse.shape[0]
            self._summation = self._summation + element_mse.sum().cpu().numpy() if self._summation is not None else element_mse.sum().cpu().numpy()

    @overrides
    def reset(self):
        """
        """
        self._item_count = 0
        self._summation = None

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[Text, float]:
        """
        """
        return_dict = {
            'score': (self._summation / self._item_count).item()
        }
        if reset:
            self.reset()

        return return_dict