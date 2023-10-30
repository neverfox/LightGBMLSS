import math
from numbers import Number
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
)
from .distribution_utils import DistributionClass
from ..utils import *


class Curry:
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.pending = args[:]
        self.kwargs = kwargs.copy(  )

    def __call__(self, *args, **kwargs):
        if kwargs and self.kwargs:
            kw = self.kwargs.copy(  )
            kw.update(kwargs)
        else:
            kw = kwargs or self.kwargs

        return self.fun(*(self.pending + args), **kw)


class Tweedie_Torch(Distribution):
    arg_constraints = {
        "loc": constraints.nonnegative,
        "scale": constraints.positive,
        "power": constraints.interval(1.0, 2.0),
    }
    support = constraints.nonnegative

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale * self.loc.pow(self.power)

    def __init__(self, loc, scale, power, validate_args=None):
        self.loc, self.scale, self.power = broadcast_all(loc, scale, power)
        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(power, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Tweedie_Torch, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.power = self.power.expand(batch_shape)
        super(Tweedie_Torch, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    @lazy_property
    def _poisson(self):
        rate = self.loc.pow(2 - self.power) / ((2 - self.power) * self.scale)
        return torch.distributions.Poisson(
            rate=rate,
            validate_args=False,
        )
    
    @lazy_property
    def _gamma(self):
        concentration = (2 - self.power) / (self.power - 1)
        scale = self.scale * (self.power - 1) * self.loc.pow(self.power - 1)
        rate = scale.pow(-1)
        return torch.distributions.Gamma(
            concentration=concentration,
            rate=rate,
            validate_args=False,
        )
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            deviates = self._poisson.sample(sample_shape=sample_shape).squeeze().detach().numpy().astype(int)
            samples = []
            for s in deviates:
                samples.append(self._gamma.sample(sample_shape=torch.Size(s)).sum().reshape(1))
            return torch.cat(samples).reshape(-1, 1)
        
    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.loc.dtype, device=self.loc.device)
        if self._validate_args:
            self._validate_sample(value)
        zeros = value == 0

        ll = torch.ones_like(value) * -(self.loc ** (2 - self.power) / (self.scale * (2 - self.power)))
        x = value[~zeros, None]
        mu = self.loc.broadcast_to(value.shape)[~zeros, None]
        phi = self.scale.broadcast_to(value.shape)[~zeros, None]
        p = self.power.broadcast_to(value.shape)[~zeros, None]

        # Quasi-likelihood
        llf = torch.log(2. * math.pi * phi) + p * torch.log(x)
        llf = llf.div(-2.)
        u = (x ** (2 - p)
                - (2 - p) * x * mu ** (1 - p)
                + (1 - p) * mu ** (2 - p))
        u *= 1. / (phi * (1 - p) * (2 - p))
        ll[~zeros] = (llf - u).squeeze()
        return ll


class Tweedie(DistributionClass):
    """
    Tweedie distribution class.

     Distributional Parameters
    --------------------------
    concentration: torch.Tensor
        shape parameter of the distribution (often referred to as alpha)
    rate: torch.Tensor
        rate = 1 / scale of the distribution (often referred to as beta)

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#gamma

    Parameters
    -------------------------
    variance_power: float
        Tweedie variance power p where 1 < p < 2. Must be provided because there is no closed-form PDF to
        compare log loss at different values of p.
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
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "exp",
                 loss_fn: str = "nll"
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please choose from 'nll' or 'crps'.")

        # Specify Response Functions
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Set the parameters specific to the distribution
        distribution = Tweedie_Torch
        param_dict = {"loc": response_fn, "scale": response_fn, "power": lambda x: sigmoid_fn(x) + 1.0}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=False,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn
                         )
