from typing import Tuple, Union
import tensorflow as tf

from dpconvcnp.utils import to_tensor

Seed = Union[tf.Tensor, Tuple[tf.Tensor], Tuple[int]]

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
    
    split = tf.random.split(seed, num=2)
    seed = split[0]
    next_seed = split[1]

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
        mean: Mean of the multivariate normal.
        cov: Covariance of the multivariate normal.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random multivariate normals with mean `loc` and covariance `scale`.
    """

    return mvn_chol(
        seed=seed,
        mean=mean,
        cov_chol=tf.linalg.cholesky(cov),
    )


def mvn_chol(
    seed: tf.Tensor,
    mean: tf.Tensor,
    cov_chol: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random multivariate normals with mean `mean` and cholesky
    factor `cov_chol`, and propagate a new random seed.

    Arguments:
        seed: Random seed for random number generator.
        mean: Mean of the multivariate normal.
        cov_chol: Cholesky of the covariance of the multivariate.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random multivariate normals with mean `loc` and covariance `scale`.
    """

    assert mean.dtype == cov_chol.dtype, "mean and chol must have the same dtype"
    assert mean.dtype in [tf.float32, tf.float64], f"Invalid dtype: {mean.dtype=}"

    # Split seed
    split = tf.random.split(seed, num=2)
    seed = split[0]
    next_seed = split[1]

    # Generate random normals
    seed, rand = randn(
        shape=tf.shape(mean),
        seed=seed,
        mean=to_tensor(0.0, mean.dtype),
        stddev=to_tensor(1.0, mean.dtype),
    )

    # Compute multiply noise by Cholesky factor and add mean
    samples = mean + tf.einsum(
        "...ij, ...j -> ...i",
        cov_chol,
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
        cov: Covariance of the multivariate normal.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random multivariate normals with mean `loc` and covariance `scale`.
    """

    # Create mean of zeroes
    mean = tf.zeros(shape=tf.shape(cov)[:-1], dtype=cov.dtype)

    return zero_mean_mvn_chol(seed=seed, cov_chol=tf.linalg.cholesky(cov))


def zero_mean_mvn_chol(
    seed: tf.Tensor,
    cov_chol: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """
    Arguments:
        seed: Random seed for random number generator.
        cov_chol: Choleksy of the covariance of the multivariate normal.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random multivariate normals with mean `loc` and covariance `scale`.
    """

    # Create mean of zeroes
    mean = tf.zeros(shape=tf.shape(cov_chol)[:-1], dtype=cov_chol.dtype)

    return mvn_chol(seed=seed, mean=mean, cov_chol=cov_chol)
