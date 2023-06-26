from typing import Tuple

import tensorflow as tf
from check_shape import check_shape

# TODO: implement SetConv Decoder

class DPSetConvEncoder(tf.Module):

    def __init__(
        self,
        *,
        points_per_unit: int,
        lenghtscale_init: float,
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

        self.log_lengthscale = tf.Variable(
            initial_value=tf.math.log(lenghtscale_init),
            trainable=True,
            dtype=dtype,
        )

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

    @property
    def lengthscale(self) -> tf.Tensor:
        return tf.exp(self.log_lengthscale)

    def call(
            self,
            x_ctx: tf.Tensor,
            y_ctx: tf.Tensor,
            x_trg: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        
        # Check context shapes
        check_shape(
            [x_ctx, y_ctx, x_trg],
            [("B", "C", "Dx"), ("B", "C", "Dy"), ("B", "T", "Dx")],
        )

        # Clip context outputs
        y_ctx = self.clip_y(y_ctx)

        # Concatenate tensor of ones to the context outputs
        y_ctx = tf.concat(
            [y_ctx, tf.ones_like(y_ctx[..., :1])],
            axis=-1,
        )  # shape (batch_size, num_ctx, Dy+1)

        # Compute endpoints of discretisation grid
        x_grid = make_discretisation_grid(
            x=tf.concat([x_ctx, x_trg], axis=1),
            points_per_unit=self.points_per_unit,
        )  # shape (batch_size, n1, ..., ndim, Dx)

        x_grid = flatten_grid(x_grid)  # shape (batch_size, num_grid_points, Dx

        # Compute matrix of weights between context points and grid points
        weights = compute_setconv_weights(
            x1=x_grid,
            x2=x_ctx,
            lengthscale=self.w_noise,
        )  # shape (batch_size, num_ctx, num_grid_points)

        # Multiply context outputs by weights
        y_grid = tf.matmul(
            weights,
            y_ctx,
        )  # shape (batch_size, num_grid_points, Dy+1)

        # TODO: add noise to y_grid

        return x_grid, y_grid


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


    def sample_noise(self, shape: tf.TensorShape) -> tf.Tensor:
        raise NotImplementedError


class SetConvDecoder(tf.Module):

    def __init__(self, *, lengthscale_init: float, trainable: bool = True):
        super().__init__()

        self.log_lengthscale = tf.Variable(
            initial_value=tf.math.log(lengthscale_init),
            trainable=trainable,
        )

    @property
    def lengthscale(self) -> tf.Tensor:
        return tf.exp(self.log_lengthscale)
    
    def call(self, x_grid: tf.Tensor, z_grid: tf.Tensor, x_trg: tf.Tensor) -> tf.Tensor:
        """Apply EQ kernel smoothing to the grid points,
        to interpolate to the target points.

        Arguments:
            x_grid: Tensor of shape (batch_size, n1, ..., ndim, Dx)
            z_grid: Tensor of shape (batch_size, n1, ..., ndim, Dz)
            x_trg: Tensor of shape (batch_size, num_trg, Dx)

        Returns:
            Tensor of shape (batch_size, num_trg, dim)
        """

        # Flatten grids
        x_grid = flatten_grid(x_grid)  # shape (batch_size, num_grid_points, Dx)
        z_grid = flatten_grid(z_grid)  # shape (batch_size, num_grid_points, Dz)

        # Compute weights
        weights = compute_setconv_weights(
            x1=x_trg,
            x2=x_grid,
            lengthscale=self.lengthscale,
        )  # shape (batch_size, num_trg, num_grid_points)

        return tf.matmul(weights, z_grid) # shape (batch_size, num_trg, Dz)
    

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
    x_mid = tf.reshape(
        x_mid,
        shape=(x.shape[0],) + (1,) * x.shape[2] + (x.shape[2],),
    )  # shape (batch_size, 1, 1, ..., 1, dim)

    # Multiply integer grid by the grid spacing and add midpoint
    grid = x_mid + grid[None, ...] / points_per_unit

    return grid


def flatten_grid(grid: tf.Tensor) -> tf.Tensor:
    """Flatten the grid tensor to a tensor of shape (batch_size, num_grid_points, dim).

    Arguments:
        grid: Tensor of shape (batch_size, n1, n2, ..., ndim, dim)

    Returns:
        Tensor of shape (batch_size, num_grid_points, dim)
    """

    return tf.reshape(grid, shape=(grid.shape[0], -1, grid.shape[-1]))


def compute_setconv_weights(
        x1: tf.Tensor,
        x2: tf.Tensor,
        lengthscale: tf.Tensor,
    ) -> tf.Tensor:
    """Compute the weights for the SetConv layer, mapping from `x1` to `x2`.

    Arguments:
        x1: Tensor of shape (batch_size, num_x1, dim)
        x2: Tensor of shape (batch_size, num_x2, dim)

    Returns:
        Tensor of shape (batch_size, num_x1, num_x2)
    """

    # Compute pairwise distances between x1 and x2
    dist2 = tf.reduce_sum(
        tf.square(x1[:, :, None, :] - x2[:, None, :, :]),
        axis=-1,
    )  # shape (batch_size, num_x1, num_x2)

    # Compute weights
    weights = tf.exp(
        -0.5 * dist2 / lengthscale**2,
    )  # shape (batch_size, num_x1, num_x2)

    return weights