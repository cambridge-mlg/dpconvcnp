import tensorflow as tf

i32 = tf.int32
i64 = tf.int64
f32 = tf.float32
f64 = tf.float64

to_tensor = lambda x, dtype: tf.convert_to_tensor(x, dtype=dtype)
cast = lambda x, dtype: tf.cast(x, dtype=dtype)

logit = lambda x: tf.math.log(x / (1.0 - x))


def expand_last_dims(tensor: tf.Tensor, ndims: int):
    return tf.reshape(
        tensor,
        shape=tf.concat([tf.shape(tensor), ndims * [1]], axis=0),
    )
