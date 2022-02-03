"""In this module we implement multiple
calibration module that could be applied
to post-hoc finetuning.
"""
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Text, Dict, Union, Optional, Callable
import torch
import torch.nn.functional as F
import functools
from dataclasses import dataclass, field
from overrides import overrides
import os
from allennlp.common.registrable import Registrable


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

    @overrides
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


@ScalingModule.register('dirichlet-calibration')
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

        self._label_dim = label_dim

        weight = torch.empty((self._label_dim, self._label_dim), dtype=torch.float32)
        bias = torch.empty((1, self._label_dim), dtype=torch.float32)

        # initializaiton
        torch.nn.init.xavier_normal_(weight)
        torch.nn.init.zeros_(bias)

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

        self._lambda = lambda_
        self._miu = miu_

    @overrides
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

    @overrides
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