import numpy as np
from scipy import special
import scipy.optimize as optim

def delta(epsilon: float, sens_per_sigma: float) -> float:
    """Compute delta for given epsilon and sensitivity per noise standard
    deviation for the Gaussian mechanism.

    Arguments:
        epsilon: DP epsilon parameter
        sens_per_sigma: Sensitivity per noise standard deviation

    Returns:
        float: Delta parameter corresponding to epsilon and sens_per_sigma.
    """

    if sens_per_sigma <= 0:
        return 0

    mu = sens_per_sigma**2 / 2

    term1 = special.erfc((epsilon - mu) / np.sqrt(mu) / 2)
    term2 = np.exp(epsilon) * special.erfc((epsilon + mu) / np.sqrt(mu) / 2)

    return 0.5 * (term1 - term2)


def find_sens_per_sigma(
        epsilon: float,
        delta_bound: float,
        lower_bound: float = 0.,
        upper_bound: float = 20.,
    ) -> float:
    """Computes the required sensitivity per noise standard deviation for
    (epsilon, delta)-DP with the Gaussian mechanism.

    Arguments:
        epsilon: DP epsilon parameter
        delta: DP delta parameter
        lower_bound: Lower bound guess on sensitivity per sigma. Defaults to 0.
        upper_bound: Upper bound guess on sensitivity per sigma. Defaults to 20.

    Returns:
        float: The required sensitivity per noise standard deviation.
    """
    return optim.brentq(
        lambda sens_per_sigma: delta(epsilon, sens_per_sigma) - delta_bound,
        a=lower_bound,
        b=upper_bound,
    )
