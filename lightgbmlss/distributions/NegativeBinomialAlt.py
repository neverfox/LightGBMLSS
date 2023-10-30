from torch.distributions import constraints
from torch.distributions import NegativeBinomial as NegativeBinomial_Torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property
from .distribution_utils import DistributionClass
from ..utils import *

class NegativeBinomialAlt_Torch(Distribution):
    r"""
    Creates a Negative Binomial distribution with an alternative parameterization.

    Args:
        rate (float or Tensor): the rate parameter
        dispersion (Tensor): the dispersion parameter
    """
    arg_constraints = {
        "rate": constraints.nonnegative,
        "dispersion1": constraints.nonnegative,
        "dispersion2": constraints.nonnegative,
    }
    support = constraints.nonnegative_integer

    def __init__(self, rate, dispersion1, dispersion2, validate_args=None):
        self.rate, self.dispersion1, self.dispersion2 = broadcast_all(rate, dispersion1, dispersion2)
        
        # TODO: change total_count to fit PyTorch parameterization
        total_count = self.rate / (self.dispersion1 - 1 + self.dispersion2 * self.rate)
        probs = 1 / (self.dispersion1 + self.dispersion2 * self.rate)

        self.base_dist = NegativeBinomial_Torch(
            total_count=total_count,
            probs=probs,
            validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        return self.base_dist.expand(batch_shape=batch_shape, _instance=_instance)

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self):
        return self.rate

    @property
    def mode(self):
        return self.rate.floor()

    @property
    def variance(self):
        return self.dispersion1 * self.rate + self.dispersion2 * self.rate ** 2

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape=sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)


class NegativeBinomialAlt(DistributionClass):
    """
    NegativeBinomial distribution class - alternative parameterization.

    Distributional Parameters
    -------------------------
    rate: torch.Tensor
        Rate parameter of the distribution.
    dispersion1: torch.Tensor
        Dispersion parameter of the distribution in the half open interval [0, 1).
    dispersion2: torch.Tensor
        Dispersion parameter of the distribution in the half open interval [0, 1).

    Source
    -------------------------
    https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/10-1831.1

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential) or "softplus" (softplus).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "exp",
                 loss_fn: str = "nll"
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")

        # Specify Response Functions
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Set the parameters specific to the distribution
        distribution = NegativeBinomialAlt_Torch
        param_dict = {"rate": response_fn, "dispersion1": response_fn, "dispersion2": response_fn}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=True,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn
                         )
