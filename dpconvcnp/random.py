from types import Tuple
import tensorflow as tf

def randint(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    minval: int,
    maxval: int,
    dtype: tf.DType = tf.int32,
) -> Tuple[tf.Tensor, tf.Tensor]:
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
) -> Tuple[tf.Tensor, tf.Tensor]:
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
    loc: float,
    scale: float,
    dtype: tf.DType = tf.float32,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Generate random normals with mean `loc` and standard deviation `scale`,
    and propagate a new random seed.

    Arguments:
        shape: Shape of the output tensor.
        seed: Random seed for random number generator.
        loc: Mean of the normal distribution.
        scale: Standard deviation of the normal distribution.
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
        loc=loc,
        scale=scale,
        dtype=dtype,
    )
