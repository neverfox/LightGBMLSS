from torch.distributions import Bernoulli as Bernoulli_Torch
from .distribution_utils import DistributionClass
from ..utils import *


class Bernoulli(DistributionClass):
    """
    Bernoulli distribution class.

    Distributional Parameters
    -------------------------
    probs: torch.Tensor
        Probability of sampling 1.

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#bernoulli

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    """
    def __init__(
        self,
        stabilization: str = "None",
        loss_fn: str = "nll",
    ):
        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError(
                "Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'."
            )
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")

        # Set the parameters specific to the distribution
        distribution = Bernoulli_Torch
        param_dict = {"probs": sigmoid_fn}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=True,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
        )
