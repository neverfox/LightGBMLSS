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
                samples.append(self._gamma.sample(sample_shape=torch.Size([s])).sum().reshape(1))
            return torch.cat(samples).reshape(-1, 1)
        
    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.loc.dtype, device=self.loc.device)
        if self._validate_args:
            self._validate_sample(value)
        zeros = value == 0

        alpha = (2 - self.power) / (1 - self.power)
        theta = self.loc ** (1 - self.power) / (1 - self.power)
        kappa = self.loc ** (2 - self.power) / (2 - self.power)
        numerator = value ** (-alpha) * (self.power - 1) ** alpha
        denominator = self.scale ** (1 - alpha) * (2 - self.power)
        z = numerator / denominator
        constant_logW = torch.log(z).max() + (1 - alpha) + alpha * torch.log(-alpha)
        jmax = value ** (2 - self.power) / (self.scale * (2 - self.power))
        j = torch.maximum(jmax.max(), torch.as_tensor(1.0))

        def _logW(alpha, j, constant_logW):
            logW = (j * (constant_logW - (1 - alpha) * torch.log(j)) -
                    math.log(2 * math.pi) - 0.5 * torch.log(-alpha) - torch.log(j))
            return logW
        
        def _logWmax(alpha, j):
            logWmax = (j * (1 - alpha) - math.log(2 * math.pi) -
                    0.5 * torch.log(-alpha) - torch.log(j))
            return logWmax

        logWmax = _logWmax(alpha, j)
        while torch.any(logWmax - _logW(alpha, j, constant_logW) < 37):
            j = j.add(1)
        j_hi = torch.ceil(j)

        j = torch.maximum(jmax.min(), torch.as_tensor(1.0))
        logWmax = _logWmax(alpha, j)

        while (torch.any(logWmax - _logW(alpha, j, constant_logW) < 37) and torch.all(j > 1)):
            j = j.sub(1)
        j_low = torch.ceil(j)

        j = torch.arange(j_low.item(), j_hi.item(), dtype=torch.float64)
        w1 = torch.tile(j, (z.shape[0], 1))

        w1 = w1.mul(torch.log(z).reshape(-1, 1))
        w1 = w1.sub(torch.special.gammaln(j + 1))
        logW = w1 - torch.special.gammaln(-alpha.reshape(-1, 1) * j)
        logWmax, _ = torch.max(logW, dim=1)
        w = torch.exp(logW - logWmax.reshape(-1, 1)).sum(dim=1)

        ll = (logWmax + torch.log(w) - torch.log(value) + (((value * theta) - kappa) / self.scale))
        ll = torch.nan_to_num(ll, neginf=-(self.loc ** (2 - self.power) / (self.scale * (2 - self.power))).item())
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
