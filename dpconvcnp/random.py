from typing import Tuple
import tensorflow as tf

Seed = Tuple[int, int]

def randint(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    minval: int,
    maxval: int,
    dtype: tf.DType = tf.int32,
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

    assert dtype in [tf.int32, tf.int64], f"Invalid dtype: {dtype}"

    seed, next_seed = tf.random.split(seed, num=2)
    return next_seed, tf.random.stateless_uniform(
        shape=shape,
        seed=seed,
        minval=minval,
        maxval=maxval,
        dtype=dtype,
    )

def randu(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    minval: float,
    maxval: float,
    dtype: tf.DType = tf.float32,
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

    assert dtype in [tf.int32, tf.int64], f"Invalid dtype: {dtype}"
    seed, next_seed = tf.random.split(seed, num=2)
    return next_seed, tf.random.stateless_uniform(
        shape=shape,
        seed=seed,
        minval=minval,
        maxval=maxval,
        dtype=dtype,
    )

def randn(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    mean: float,
    stddev: float,
    dtype: tf.DType = tf.float32,
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

    assert dtype in [tf.float32, tf.float64], f"Invalid dtype: {dtype}"
    
    seed, next_seed = tf.random.split(seed, num=2)
    return next_seed, tf.random.stateless_normal(
        shape=shape,
        seed=seed,
        mean=mean,
        stddev=stddev,
        dtype=dtype,
    )

def mvn(
    seed: tf.Tensor,
    mean: float,
    cov: float,
    dtype: tf.DType = tf.float32,
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

    assert dtype in [tf.float32, tf.float64], f"Invalid dtype: {dtype}"

    # Split seed
    seed, next_seed = tf.random.split(seed, num=2)

    # Generate random normals
    seed, rand = randn(
        shape=mean.shape,
        seed=seed,
        mean=0.0,
        stddev=1.0,
        dtype=dtype,
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
    cov: float,
    dtype: tf.DType = tf.float32,
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
    mean = tf.zeros(shape=cov.shape[:-1], dtype=dtype)

    return mvn(
        seed=seed,
        mean=mean,
        cov=cov,
        dtype=dtype,
    )