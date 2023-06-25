import tensorflow as tf
from check_shape import check_shape


class DPSetConv(tf.Module):

    def __init__(
        self,
        *,
        points_per_unit: int,
        y_bound_init: float,
        w_noise_init: float,
        y_bound_trainable: bool = True,
        w_noise_trainable: bool = True,
        dtype: tf.DType = tf.float32,
        name="dp_set_conv",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.points_per_unit = points_per_unit

        self.y_bound = tf.Variable(
            initial_value=y_bound_init,
            trainable=y_bound_trainable,
            dtype=dtype,
        )

        self.y_bound = tf.Variable(
            initial_value=w_noise_init,
            trainable=w_noise_trainable,
            dtype=dtype,
        )

    def call(self, x_ctx: tf.Tensor, y_ctx: tf.Tensor, x_trg: tf.Tensor) -> tf.Tensor:
        
        # Check context shapes
        check_shape(
            [x_ctx, y_ctx, x_trg],
            [("B", "C", "Dx"), ("B", "C", "Dy"), ("B", "T", "Dx")],
        )

        # Compute endpoints of discretisation grid
        x_grid = make_discretisation_grid(
            x=tf.concat([x_ctx, x_trg], axis=1),
            points_per_unit=self.points_per_unit,
        )



    def clip_y(self, y_ctx: tf.Tensor) -> tf.Tensor:
        """Clip the context outputs to be within the range [-y_bound, y_bound].

        Arguments:
            y_ctx: Tensor of shape (batch_size, num_ctx, dim) containing the
                context outputs.
            
        Returns:
            Tensor of shape (batch_size, num_ctx, dim) containing the clipped
                context outputs.
        """

        return tf.clip_by_value(y_ctx, -self.y_bound, self.y_bound)


def make_discretisation_grid(
        x: tf.Tensor,
        points_per_unit: int,
        margin: float,
) -> tf.Tensor:
    """Create a grid with density `points_per_unit` in each dimension,
    such that the grid contains all the points in `x` and has a margin of
    at least `margin` around the points in `x`, and is centered at the
    midpoint of the points in `x`.

    Arguments:
        x: Tensor of shape (batch_size, num_points, dim) containing the
            points.
        points_per_unit: Number of points per unit length in each dimension.
        margin: Margin around the points in `x`.

    Returns:
        Tensor of shape (batch_size, n1, n2, ..., ndim, dim)
    """

    # Compute min and max of each dimension
    x_min = tf.reduce_min(x, axis=1)  # shape (batch_size, dim)
    x_max = tf.reduce_max(x, axis=1)  # shape (batch_size, dim)

    # Compute half the number of points in each dimension
    num_half_points = tf.math.ceil(
        (0.5 * (x_max - x_min) + margin) * points_per_unit
    )  # shape (batch_size, dim)

    # Take the maximum over the batch, in order to use the same number of 
    # points across all tasks in the batch, in order to enable tensor batching
    num_half_points = tf.reduce_max(num_half_points, axis=0)  # shape (dim,)

    # Compute the discretisation grid
    grid = tf.stack(
        tf.meshgrid(
            *[tf.range(-num, num+1, dtype=x.dtype) for num in num_half_points]
        ),
        axis=-1,
    )  # shape (n1, n2, ..., ndim, dim)

    # Compute midpoints of each dimeension and expand the midpoint tensor
    # to match the number of dimensions in the grid
    x_mid = 0.5 * (x_min + x_max)  # shape (batch_size, dim)
    for _ in range(grid.ndim - x_mid.ndim + 1):
        x_mid = tf.expand_dims(x_mid, axis=1)  # shape (batch_size, 1, 1, ..., 1, dim)

    # Multiply integer grid by the grid spacing and add midpoint
    grid = x_mid + grid[None, ...] / points_per_unit

    return tf.reshape(grid, (x.shape[0], -1, x.shape[2]))
