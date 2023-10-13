from .zero_inflated import ZeroAdjustedInverseGaussian as ZeroAdjustedInverseGaussian_Torch
from .distribution_utils import DistributionClass
from ..utils import *


class ZAIG(DistributionClass):
    """
    Zero-Adjusted Inverse Gaussian distribution class.

    The zero-adjusted Inverse Gaussian distribution is similar to the Inverse Gaussian distribution but allows zeros as y values.

    Distributional Parameters
    -------------------------
    loc: torch.Tensor
        Mean of the distribution.
    scale: torch.Tensor
        Standard deviation of the distribution.
    gate: torch.Tensor
        Probability of zeros given via a Bernoulli distribution.

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
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")

        # Specify Response Functions
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'."
            )

        # Set the parameters specific to the distribution
        distribution = ZeroAdjustedInverseGaussian_Torch
        param_dict = {
            "loc": response_fn,
            "scale": response_fn,
            "gate": sigmoid_fn,
        }
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
