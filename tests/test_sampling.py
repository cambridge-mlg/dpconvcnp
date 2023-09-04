import pytest
import tensorflow as tf
import numpy as np

from dpconvcnp.random import Seed, zero_mean_mvn_on_grid_from_chol


def eq(x: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    assert len(tf.shape(x)) == 1

    return tf.exp(
        -0.5 * ((x[:, None] - x[None, :]) / scale) ** 2
    ) + 1e-6 * tf.eye(tf.shape(x)[-1])


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("seed", range(10))
def test_zero_mean_mvn_on_grid_from_chol_covariance(ndim: int, seed: int):

    # Set test constants
    n_samples = 16384
    scale = 0.5
    num_points_dw = 10
    seed = [0, seed]

    # Dimensionwise grid coordinates
    x_dw = [tf.linspace(-1.0, 1.0, num_points_dw) for _ in range(ndim)]

    # Dimensionwise covariance matrices and cholesky factors
    k_dw = [eq(x, scale=scale) for x in x_dw]
    chol_dw = [tf.linalg.cholesky(k) for k in k_dw]
    chol_dw = [tf.tile(chol[None], [n_samples, 1, 1]) for chol in chol_dw]

    # Sample from zero mean multivariate normal
    _, noise = zero_mean_mvn_on_grid_from_chol(seed, chol_dw)

    # Compute sample covariance
    noise1 = noise[:]
    noise2 = noise[:]

    for _ in range(ndim):
        noise1 = tf.expand_dims(noise1, axis=-1)
        noise2 = tf.expand_dims(noise2, axis=1)

    # Permute entries of k_sample
    k_sample = tf.reduce_mean(noise1 * noise2, axis=0)

    # Compute exact covariance
    k_exact = k_dw[0].numpy()
    for k in k_dw[1:]:
        k_exact = np.multiply.outer(k_exact, k.numpy())

    # Permute entries of k_exact if needed
    if ndim > 1:
        k_exact = np.transpose(
            k_exact,
            axes=[j*ndim + i for i in range(ndim) for j in range(ndim)],
        )
    k_exact = tf.convert_to_tensor(k_exact, dtype=tf.float32)

    # Compute error and check
    error = tf.reduce_mean(tf.abs(k_sample - k_exact))
    assert error < 0.02, f"{error=}"
