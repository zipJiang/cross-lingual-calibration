"""In this module we implement multiple
calibration module that could be applied
to post-hoc finetuning.
"""
import json
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Text, Dict, Union, Optional, Callable
from sklearn.metrics import mean_poisson_deviance
import torch
import torch.nn.functional as F
import functools
from dataclasses import dataclass, field
from overrides import overrides
import os
from allennlp.common.registrable import Registrable
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.data_loaders import DataLoader
from ...svgp.gp_classes import FlexibleNumClassSoftmaxLikelihood, ApproximateGpCalibrationModel
from gpytorch.mlls import VariationalELBO


@dataclass
class ScalingOutput:
    """Keeping the output from the Scaling Modules
    """
    original_logits: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor = field(default=None, metadata={
        'help': 'Only returns this when label is provided.'
    })


class ScalingModule(Registrable, torch.nn.Module):
    """Abstract class for the scaling module
    """
    def __init__(self):
        """This is a abstract class that should
        be used to construct Scaling Modules.
        """
        super().__init__()


@ScalingModule.register('platt-scaling')
class PlattScaling(ScalingModule):
    def __init__(self, label_dim: int,
                 vector_scaling: Optional[bool] = True):
        """This function is just a linear transformation,
        where the input will be the logits of the data.

        vector_scaling: bool, set to True to get the
        scaling with diagnal matrices.
        """
        super().__init__()

        self._label_dim = label_dim
        self._vector_scaling = vector_scaling

        if not self._vector_scaling:
            weight = torch.empty((self._label_dim, self._label_dim), dtype=torch.float32)
            bias = torch.empty((1, self._label_dim), dtype=torch.float32)
            torch.nn.init.xavier_normal_(weight)
        else:
            weight = torch.empty((self._label_dim,), dtype=torch.float32)
            bias = torch.empty((1, self._label_dim), dtype=torch.float32)
            torch.nn.init.uniform_(weight)

        # initializaiton
        torch.nn.init.zeros_(bias)

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

    def forward(
        self,
        input_: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> ScalingOutput:
        """This is the scaling function that depends on
        the original logits.

        input: --- [batch_size, label_dim] the original predicted logits
        label --- [batch_size] the label for the correct prediction
        """

        if self._vector_scaling:
            weight = torch.diag_embed(self.weight)
        else:
            weight = self.weight

        logits = input_ @ weight + self.bias

        if label is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            loss_ = loss_func(logits, label)
        else:
            loss_ = None

        return_val = ScalingOutput(original_logits=input_,
                                    logits=logits,
                                    loss=loss_)

        return return_val


@ScalingModule.register('dirichlet-calibration', constructor="from_partial_object")
class DirichletCalibration(ScalingModule):
    def __init__(self, label_dim: int,
                 lambda_: Optional[float] = None,
                 miu_: Optional[float] = None):
        """This is largely the same as matrix scaling,
        but we implement this with a log_sofmax transformation
        and regularization.

        lambda_: float - the coefficient for the off-diagnal regularization
        miu_: float - the coefficient for the bias-term
        """
        super().__init__()

        # self._label_dim = label_dim

        weight = torch.empty((self._label_dim, self._label_dim), dtype=torch.float32)
        bias = torch.empty((1, self._label_dim), dtype=torch.float32)

        # initializaiton
        torch.nn.init.xavier_normal_(weight)
        torch.nn.init.zeros_(bias)

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

        self._lambda = lambda_
        self._miu = miu_
        
    @classmethod
    def from_partial_object(
        cls,
        data_loader: DataLoader,
        lambda_: Optional[float] = None,
        miu_: Optional[float] = None
    ) -> "DirichletCalibration":
        """
        """
        data_loader.index_with(Vocabulary())
        for batch in data_loader:
            label_dim = batch['logits'].size(-1)
            break

        return cls(
            label_dim=label_dim,
            lambda_=lambda_,
            miu_=miu_
        )

    def forward(self, input_: torch.Tensor, label: Optional[torch.Tensor] = None) -> ScalingOutput:
        """This is the scaling function that depends on
        the original logits.

        input: --- [batch_size, label_dim] the original predicted logits
        label --- [batch_size] the label for the correct prediction
        """
        input_ = F.log_softmax(input_, dim=-1)  # The main vibe for dirichlet calibration

        logits = input_ @ self.weight + self.bias

        if label is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            loss_ = loss_func(logits, label)

            if self._lambda is not None:
                # calculate the off diagnal regularier
                squared_weight = torch.square(self.weight)
                loss_ += self._lambda * (torch.sum(squared_weight) - torch.sum(torch.diag(squared_weight, diagonal=0)))

            if self._miu is not None:
                # calculate the bias normalizing term
                squared_bias = torch.square(self.bias)
                loss_ += self._miu * torch.sum(squared_bias)

        else:
            loss_ = None

        return_val = ScalingOutput(original_logits=input_,
                                   logits=logits,
                                   loss=loss_)

        return return_val


@ScalingModule.register('temperature-scaling')
class TemperatureScaling(ScalingModule):
    def __init__(self):
        """
        """
        super().__init__()

        weight = torch.tensor([[1.]], dtype=torch.float32)
        self.weight = torch.nn.Parameter(weight)

    def forward(self,
        input_: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> ScalingOutput:
        """
        input_: --- [batch, label_dim]
        """

        logits = input_ * self.weight

        if label is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            loss_ = loss_func(logits, label)
        else:
            loss_ = None

        return_val = ScalingOutput(
            original_logits=input_,
            logits=logits,
            loss=loss_
        )

        return return_val
    
    
@ScalingModule.register("beta-calibration")
class BetaCalibration(ScalingModule):
    """This module scaled the probability with the
    beta family.
    """
    def __init__(
        self,
        epsilon: float = 2e-8
    ):
        """
        """
        super().__init__()
        
        weight = torch.tensor([[-1.], [1.]], dtype=torch.float32)
        bias = torch.tensor(0., dtype=torch.float32)

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        self.epsilon = epsilon

    def forward(
        self,
        input_: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> ScalingOutput:
        """Beta calibration can only be applied to
        binary calibration, so that we also need to
        group input_ logits according to their
        probability.
        """
        
        inv_link = torch.nn.Softmax(dim=-1)
        probs = inv_link(input_)
        predictions = torch.argmax(probs, dim=-1, keepdim=True)
        
        # construct binary classification logits
        positives = torch.gather(probs, dim=-1, index=predictions)
        
        binary_probs = torch.cat((1 - positives, positives), dim=-1)
        binary_probs[binary_probs < self.epsilon] = self.epsilon
        binary_probs[binary_probs > 1. - self.epsilon] = 1. - self.epsilon
        log_binary_probs = binary_probs.log()
        # log_binary_probs[log_binary_probs < self._min_float] = self._min_float
        
        # back generate reversed probabilities logits
        pred_logits = (log_binary_probs @ self.weight + self.bias)  # [batch_size, 1]
        
        reconstructed_probs = torch.scatter(
            input=((1 - torch.sigmoid(pred_logits)) / (input_.size(-1) - 1.)).repeat(1, input_.size(-1)),
            dim=-1,
            index=predictions,
            src=torch.sigmoid(pred_logits)
        )
        
        logits = (reconstructed_probs / torch.sum(reconstructed_probs, dim=-1, keepdim=True)).log()
        
        if label is not None:
            # can calculate logistic regression loss
            loss_func = torch.nn.BCEWithLogitsLoss()
            
            # calculate the new labels {0: wrong, 1: right}
            binary_labels = (predictions[:, 0] == label).float()
            
            # calculate binary CE loss
            loss = loss_func(pred_logits.flatten(), binary_labels)
            
            return ScalingOutput(
                original_logits=input_,
                logits=logits,
                loss=loss
            )
        else:
            return ScalingOutput(
                original_logits=input_,
                logits=logits
            )

        
@ScalingModule.register("gp-calibration", constructor='from_partial_object')
class GaussianProcessScaling(ScalingModule):
    """This module scales the probability with
    a scaling module that is a shared gaussian process
    over all logits
    """
    def __init__(
        self,
        inducing_points: torch.Tensor,
        num_data: int,
        min_float: -2e3,
        mean_init_std: Optional[float] = 1e-3,
        is_diag: Optional[bool] = False
    ):
        """
        """
        super().__init__()
        self._calibration_model = ApproximateGpCalibrationModel(
            inducing_points=inducing_points,
            mean_init_std=mean_init_std,
            is_diag=is_diag
        )
        
        self._likelihood = FlexibleNumClassSoftmaxLikelihood()
        
        self._loss_func = VariationalELBO(
            likelihood=self._likelihood,
            model=self._calibration_model,
            num_data=num_data
        )
        
        self._min_float = min_float
        
    @classmethod
    def from_partial_object(
        cls,
        data_path: Text,
        logits_key: Text = 'logit',
        num_inducing_points: Optional[int] = 10,
        mean_init_std: Optional[float] = 1e-2,
        min_float: float = -2e2,
        is_diag: bool = False
    ) -> "GaussianProcessScaling":
        """This is a calibration_module constructor that
        takes in a dataloader.
        """

        import numpy as np
        from scipy.cluster.vq import kmeans
        
        with open(data_path, 'r', encoding='utf-8') as file_:
            screening = []
            for line in file_:
                screening.extend([max(k, min_float) for k in json.loads(line)[logits_key]])
            num_data = len(screening)

        # Z = kmeans(
        #     obs=np.array(screening).reshape(-1, 1),
        #     k_or_guess=min(num_data, num_inducing_points)
        # )[0]
        Z = 3 * torch.randn(num_inducing_points, 1)
        
        return cls(
            inducing_points=torch.tensor(Z, dtype=torch.float32),
            num_data=num_data,
            min_float=min_float,
            mean_init_std=mean_init_std,
            is_diag=is_diag
        )
        
    def forward(self,
        input_: torch.Tensor,
        label: Optional[torch.Tensor] = None
    ) -> ScalingOutput:
        """
        
        input_: is of shape [batch_size, num_labels]
        label: is of shape [batch_size, num_labels]
        """

        # flatten input_ first
        input_[input_ < self._min_float] = self._min_float
        X = input_.unsqueeze(-1)  # [batch_size, num_classes, 1]
        
        f_dist = self._calibration_model(X)
        
        pred_logits = torch.mean(self._likelihood(f_dist).logits, dim=0)
        
        if label is not None:
            loss = -self._loss_func(f_dist, label)
            
            return ScalingOutput(
                original_logits=input_,
                logits=pred_logits,
                loss=loss
            )
            
        return ScalingOutput(
            origial_logits=input_,
            logits=pred_logits
        )