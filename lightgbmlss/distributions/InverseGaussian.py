import math
from numbers import Number, Real

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

from .distribution_utils import DistributionClass
from ..utils import *


class InverseGaussian_Torch(ExponentialFamily):
    r"""
    Creates an Inverse Gaussian (also called Wald) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = InverseGaussian(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Inverse Gaussian distributed with loc=1 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): scale of the distribution (often referred to as lambda)
    """
    arg_constraints = {"loc": constraints.positive, "scale": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc * (
            (1 + 9 * self.loc.square() * (4 * self.scale.square()).reciprocal()).sqrt()
            - 3 * self.loc * (2 * self.scale).reciprocal()
        )

    @property
    def variance(self):
        return self.loc.pow(3) * self.scale.reciprocal()

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGaussian_Torch, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(InverseGaussian_Torch, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        v = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        y = v.square()
        x = (
            self.loc
            + (self.loc.square() * y) / (2 * self.scale)
            - (
                self.loc
                / (2 * self.scale)
                * (4 * self.loc * self.scale * y + self.loc * self.loc * y * y).sqrt()
            )
        )
        z = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return torch.where(self.loc / (self.loc + x) >= z, x, self.loc.square() / x)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # scale the value
        x = value / self.scale
        log_scale = self.scale.log()
        return (
            -0.5 * math.log(2 * math.pi)
            - 1.5 * torch.log(x)
            - ((x - self.loc) / self.loc) ** 2 / (2 * x)
        ) - log_scale

    """
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        x1 = torch.sqrt(self.scale / value) * (value * self.loc.reciprocal() - 1)
        x2 = torch.neg(
            torch.sqrt(self.scale / value) * (value * self.loc.reciprocal() + 1)
        )
        return (
            0.5
            * (1 + torch.erf((x1 - self.loc) * self.scale.reciprocal() / math.sqrt(2)))
            + torch.exp(2 * self.scale * self.loc.reciprocal())
            + 0.5
            * (1 + torch.erf((x2 - self.loc) * self.scale.reciprocal() / math.sqrt(2)))
        )
    """


class InverseGaussian:
    """
    Inverse Gaussian distribution class.

    Distributional Parameters
    -------------------------
    loc: torch.Tensor
        Mean of the distribution (often referred to as mu).
    scale: torch.Tensor
        Scale of the distribution (often referred to as lambda).

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential) or "softplus" (softplus).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    """

    def __init__(
        self,
        stabilization: str = "None",
        response_fn: str = "exp",
        loss_fn: str = "nll",
    ):
        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError(
                "Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'."
            )
        if loss_fn not in ["nll", "crps"]:
            raise ValueError(
                "Invalid loss function. Please choose from 'nll' or 'crps'."
            )

        # Specify Response Functions
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'."
            )

        # Set the parameters specific to the distribution
        distribution = InverseGaussian_Torch
        param_dict = {"loc": response_fn, "scale": response_fn}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
        )
