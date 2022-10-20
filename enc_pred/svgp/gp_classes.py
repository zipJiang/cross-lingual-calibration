"""Define our own gp_class components
that we try to use in the final optimization.
"""
import warnings
import gpytorch
from gpytorch.likelihoods import Likelihood, SoftmaxLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal, base_distributions, Distribution
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.models import ApproximateGP
from gpytorch.kernels import Kernel
from typing import Callable
from sklearn.model_selection import learning_curve

from overrides import overrides
import torch


class FlexibleNumClassSoftmaxLikelihood(Likelihood):
    """
    """
    def __init__(
        self, 
    ):
        super().__init__()

    def forward(self, function_samples, *params, **kwargs):
        num_data, num_features = function_samples.shape[-2:]

        mixed_fs = function_samples
        res = base_distributions.Categorical(logits=mixed_fs)
        return res

    def __call__(self, function, *params, **kwargs):
        # if isinstance(function, Distribution) and not isinstance(function, MultitaskMultivariateNormal):
            # disable the original warnings
            # function = MultitaskMultivariateNormal.from_batch_mvn(function)

        return super().__call__(function, *params, **kwargs)

        
class ApproximateGpCalibrationModel(ApproximateGP):
    """Our approximate GP, that produce the function sampling
    with restricted output.
    """
    def __init__(
        self,
        inducing_points: torch.Tensor = None,
        mean_init_std: float = 1e-3,
        lengthscale_prior: gpytorch.priors.Prior = gpytorch.priors.NormalPrior(loc=10., scale=1e-3),
        is_diag: bool = False,
    ):
        if not is_diag:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.size(0),
                mean_init_std=mean_init_std
            )
            
            variational_strategy = VariationalStrategy(
                model=self,
                inducing_points=inducing_points,
                variational_distribution=variational_distribution,
                learn_inducing_locations=True
            )
        else:
            # if is_diag, we use the single diag prior
            pass

        super().__init__(
            variational_strategy=variational_strategy
        )

        # we use this default seems that this is not a parameter in the original implementation
        self.mean_module = gpytorch.means.ConstantMean()

        # TODO: current differences: we do not apply white noice to self-kernel calculation.
        self.covar_module = gpytorch.kernels.RBFKernel(
            lengthscale_prior=lengthscale_prior
        )
        
    @overrides
    def forward(self, x: torch.Tensor):
        """x should be of shape: (num_data, num_class + M, 1)
        """

        mean = self.mean_module(x)
        covar = self.covar_module(x)
        
        # return the not-sampled output here.
        return MultivariateNormal(
            mean=mean,
            covariance_matrix=covar
        )