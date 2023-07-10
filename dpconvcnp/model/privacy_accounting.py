import tensorflow as tf
import tensorflow_probability as tfp

from dpconvcnp.utils import to_tensor

import numpy as np
from scipy import special
import scipy.optimize as optim


def numpy_delta_from_epsilon_and_sens_per_sigma(epsilon: float, sens_per_sigma: float) -> float:
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


def numpy_sens_per_sigma(
        epsilon: float,
        delta: float,
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
        lambda sens_per_sigma: numpy_delta_from_epsilon_and_sens_per_sigma(
            epsilon,
            sens_per_sigma
        ) - delta,
        a=lower_bound,
        b=upper_bound,
    )


def delta_from_epsilon_and_sens_per_sigma(epsilon: tf.Tensor, sens_per_sigma: tf.Tensor) -> tf.Tensor:
    """Compute delta for given epsilon and sensitivity per noise standard
    deviation for the Gaussian mechanism.

    Arguments:
        epsilon: DP epsilon parameter
        sens_per_sigma: Sensitivity per noise standard deviation

    Returns:
        float: Delta parameter corresponding to epsilon and sens_per_sigma.
    """

    mu = sens_per_sigma**2 / 2

    term1 = tf.math.erfc((epsilon - mu) / tf.sqrt(mu) / 2)
    term2 = tf.exp(epsilon) * tf.math.erfc((epsilon + mu) / tf.sqrt(mu) / 2)

    return 0.5 * (term1 - term2)


def sens_per_sigma(
        epsilon: tf.Tensor,
        delta: tf.Tensor,
        lower_bound: tf.Tensor = 1e-2,
        upper_bound: tf.Tensor = 10.,
    ) -> tf.Tensor:
    """Computes the required sensitivity per noise standard deviation for
    (epsilon, delta)-DP with the Gaussian mechanism.

    Arguments:
        epsilon: DP epsilon parameter
        delta: DP delta parameter
        lower_bound: Lower bound guess on sensitivity per sigma. Defaults to 0.
        upper_bound: Upper bound guess on sensitivity per sigma. Defaults to 10.

    Returns:
        float: The required sensitivity per noise standard deviation.
    """
    return tfp.math.find_root_chandrupatla(
        lambda sens_per_sigma: delta_from_epsilon_and_sens_per_sigma(epsilon, sens_per_sigma) - delta,
        low=to_tensor(lower_bound, dtype=epsilon.dtype),
        high=to_tensor(upper_bound, dtype=epsilon.dtype),
    ).estimated_root