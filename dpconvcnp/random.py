from typing import Tuple
import tensorflow as tf

from dpconvcnp.types import Seed
from dpconvcnp.utils import f32, f64, to_tensor

def randint(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    minval: tf.Tensor,
    maxval: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random integers in the range `[minval, maxval]`,
    uniformly distributed, and propagate a new random seed.

    Arguments:
        shape: Shape of the output tensor.
        seed: Random seed for random number generator.
        minval: Lower bound of the range of random integers to generate.
        maxval: Upper bound of the range of random integers to generate.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random integers in the range `[minval, maxval]`.
    """

    assert minval.dtype == maxval.dtype, "minval and maxval must have the same dtype"
    assert minval.dtype in [tf.int32, tf.int64], f"Invalid dtype: {minval.dtype=}"

    seed, next_seed = tf.random.split(seed, num=2)
    return next_seed, tf.random.stateless_uniform(
        shape=shape,
        seed=seed,
        minval=minval,
        maxval=maxval+1,
        dtype=minval.dtype,
    )

def randu(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    minval: tf.Tensor,
    maxval: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random uniforms in the range `[minval, maxval]`,
    uniformly distributed, and propagate a new random seed.

    Arguments:
        shape: Shape of the output tensor.
        seed: Random seed for random number generator.
        minval: Lower bound of the range of random uniforms to generate.
        maxval: Upper bound of the range of random uniforms to generate.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random uniforms in the range `[minval, maxval]`.
    """

    assert minval.dtype == maxval.dtype, "minval and maxval must have the same dtype"
    assert minval.dtype in [tf.float32, tf.float64], f"Invalid dtype: {minval.dtype=}"
    
    seed, next_seed = tf.random.split(seed, num=2)
    return next_seed, tf.random.stateless_uniform(
        shape=shape,
        seed=seed,
        minval=minval,
        maxval=maxval,
        dtype=minval.dtype,
    )

def randn(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    mean: tf.Tensor,
    stddev: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random normals with mean `mean` and standard deviation `stddev`,
    and propagate a new random seed.

    Arguments:
        shape: Shape of the output tensor.
        seed: Random seed for random number generator.
        mean: Mean of the normal distribution.
        stddev: Standard deviation of the normal distribution.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random normals with mean `loc` and standard deviation `scale`.
    """

    assert mean.dtype == stddev.dtype, "mean and stddev must have the same dtype"
    assert mean.dtype in [tf.float32, tf.float64], f"Invalid dtype: {mean.dtype=}"
    
    seed, next_seed = tf.random.split(seed, num=2)

    return next_seed, tf.random.stateless_normal(
        shape=shape,
        seed=seed,
        mean=mean,
        stddev=stddev,
        dtype=mean.dtype,
    )

def mvn(
    seed: tf.Tensor,
    mean: tf.Tensor,
    cov: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random multivariate normals with mean `mean` and covariance `cov`,
    and propagate a new random seed.

    Arguments:
        seed: Random seed for random number generator.
        mean: Mean of the multivariate normal distribution.
        cov: Covariance of the multivariate normal distribution.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random multivariate normals with mean `loc` and covariance `scale`.
    """

    assert mean.dtype == cov.dtype, "mean and cov must have the same dtype"
    assert mean.dtype in [tf.float32, tf.float64], f"Invalid dtype: {mean.dtype=}"

    # Split seed
    seed, next_seed = tf.random.split(seed, num=2)

    # Generate random normals
    seed, rand = randn(
        shape=mean.shape,
        seed=seed,
        mean=to_tensor(0.0, mean.dtype),
        stddev=to_tensor(1.0, mean.dtype),
    )

    # Compute multiply noise by Cholesky factor and add mean
    samples = mean + tf.einsum(
        "...ij, ...j -> ...i",
        tf.linalg.cholesky(cov),
        rand,
    )

    return next_seed, samples

def zero_mean_mvn(
    seed: tf.Tensor,
    cov: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random multivariate normals with mean zero and covariance `cov`,
    and propagate a new random seed.

    Arguments:
        seed: Random seed for random number generator.
        cov: Covariance of the multivariate normal distribution.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random multivariate normals with mean `loc` and covariance `scale`.
    """

    # Create mean of zeroes
    mean = tf.zeros(shape=cov.shape[:-1], dtype=cov.dtype)

    return mvn(seed=seed, mean=mean, cov=cov)