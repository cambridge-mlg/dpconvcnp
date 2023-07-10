from typing import Tuple

import tensorflow as tf
from check_shape import check_shape

from dpconvcnp.random import zero_mean_mvn_chol
from dpconvcnp.random import Seed
from dpconvcnp.utils import f64, cast, logit
from dpconvcnp.model.privacy_accounting import (
    sens_per_sigma as dp_sens_per_sigma,
    numpy_sens_per_sigma as dp_numpy_sens_per_sigma,
)


class DPSetConvEncoder(tf.Module):

    def __init__(
        self,
        *,
        points_per_unit: int,
        lenghtscale_init: float,
        y_bound_init: float,
        w_noise_init: float,
        margin: float,
        lengthscale_trainable: bool = True,
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
            trainable=lengthscale_trainable,
            dtype=dtype,
        )

        self.log_y_bound = tf.Variable(
            initial_value=tf.math.log(y_bound_init),
            trainable=y_bound_trainable,
            dtype=dtype,
        )

        self.logit_w_noise = tf.Variable(
            initial_value=logit(w_noise_init),
            trainable=w_noise_trainable,
            dtype=dtype,
        )

        self.margin = margin

    @property
    def lengthscale(self) -> tf.Tensor:
        return tf.exp(self.log_lengthscale)

    @property
    def y_bound(self) -> tf.Tensor:
        return tf.exp(self.log_y_bound) + 1e-2
    
    @property
    def w_noise(self) -> tf.Tensor:
        return (1 - 2e-2) * tf.nn.sigmoid(self.logit_w_noise) + 1e-2
    
    def data_sigma(self, sens_per_sigma: tf.Tensor) -> tf.Tensor:
        return 2.0 * self.y_bound / (sens_per_sigma * self.w_noise ** 0.5)

    def density_sigma(self, sens_per_sigma: tf.Tensor) -> tf.Tensor:
        return 2**0.5 / (sens_per_sigma * (1 - self.w_noise) ** 0.5)

    def __call__(
            self,
            seed: Seed,
            x_ctx: tf.Tensor,
            y_ctx: tf.Tensor,
            x_trg: tf.Tensor,
            epsilon: tf.Tensor,
            delta: tf.Tensor,
        ) -> Tuple[Seed, tf.Tensor, tf.Tensor]:
        
        # Check context shapes
        check_shape(
            [x_ctx, y_ctx, x_trg],
            [("B", "C", "Dx"), ("B", "C", 1), ("B", "T", "Dx")],
        )

        # Clip context outputs and concatenate tensor of ones
        y_ctx = self.clip_y(y_ctx)  # shape (batch_size, num_ctx, 1)
        y_ctx = tf.concat(
            [y_ctx, tf.ones_like(y_ctx)],
            axis=-1,
        )  # shape (batch_size, num_ctx, Dy+1)

        # Compute endpoints of discretisation grid
        x_grid = make_discretisation_grid(
            x=tf.concat([x_ctx, x_trg], axis=1),
            points_per_unit=self.points_per_unit,
            margin=self.margin,
        )  # shape (batch_size, n1, ..., ndim, Dx)
        x_grid_flat = flatten_grid(x_grid)  # shape (batch_size, num_grid_points, Dx)

        # Compute matrix of weights between context points and grid points
        weights = compute_eq_weights(
            x1=x_grid_flat,
            x2=x_ctx,
            lengthscale=self.lengthscale,
        )  # shape (batch_size, num_ctx, num_grid_points)

        # Multiply context outputs by weights
        z_grid_flat = tf.matmul(
            weights,
            y_ctx,
        )  # shape (batch_size, num_grid_points, 2)

        # Sample noise and add it to the data and density channels
        seed, noise = self.sample_noise(
            seed=seed,
            x_grid=x_grid_flat,
            epsilon=epsilon,
            delta=delta,
        )
        
        check_shape(
            [z_grid_flat, noise],
            [("B", "G", 2), ("B", "G", 2)],
        )

        z_grid_flat = z_grid_flat + noise

        # Reshape grid
        z_grid = tf.reshape(
            z_grid_flat,
            shape=(x_grid.shape[0],) + x_grid.shape[1:-1] + (z_grid_flat.shape[-1],),
        )  # shape (batch_size, n1, ..., ndim, 2)

        return seed, x_grid, z_grid


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


    def sample_noise(
            self,
            seed: Seed,
            x_grid: tf.Tensor,
            epsilon: tf.Tensor,
            delta: tf.Tensor,
        ) -> tf.Tensor:
        """Sample noise for the density and data channels.

        Arguments:
            x_grid: Tensor of shape (batch_size, num_grid_points, Dx)
            epsilon: DP epsilon parameter
            delta: DP delta parameter

        Returns:
            Tensor of shape (batch_size, num_grid_points, 2)
        """

        # Save input data type
        in_dtype = x_grid.dtype

        # Convert everything to float64 for numerical accuracy
        x_grid = cast(x_grid, dtype=f64)
        epsilon = cast(epsilon, dtype=f64)
        delta = cast(delta, dtype=f64)

        # Compute noise GP covariance and its cholesky
        kxx = compute_eq_weights(
            x1=x_grid,
            x2=x_grid,
            lengthscale=cast(self.lengthscale, dtype=f64),
        )
        kxx_chol = tf.linalg.cholesky(
            kxx + tf.eye(kxx.shape[-1], dtype=kxx.dtype) * 1e-6
        )

        # Draw noise samples for the density and data channels
        seed, data_noise = zero_mean_mvn_chol(
            seed=seed,
            cov_chol=kxx_chol,
        )  # shape (batch_size, num_grid_points,)

        seed, density_noise = zero_mean_mvn_chol(
            seed=seed,
            cov_chol=kxx_chol,
        )  # shape (batch_size, num_grid_points,) 

        # Compute sensitivity per sigma
        sens_per_sigma = dp_sens_per_sigma(epsilon=epsilon, delta=delta)

        # Convert back to input data types
        data_noise = cast(data_noise, dtype=in_dtype)
        density_noise = cast(density_noise, dtype=in_dtype)
        sens_per_sigma = cast(sens_per_sigma, dtype=in_dtype)

        # Multiply noise by standard deviations
        data_noise = data_noise * self.data_sigma(
            sens_per_sigma=sens_per_sigma,
        )[:, None]

        density_noise = density_noise * self.density_sigma(
            sens_per_sigma=sens_per_sigma,
        )[:, None]

        return seed, tf.stack(
            [data_noise, density_noise],
            axis=-1,
        )  # shape (batch_size, num_grid_points, 2)


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
    
    def __call__(
            self,
            x_grid: tf.Tensor,
            z_grid: tf.Tensor,
            x_trg: tf.Tensor,
        ) -> tf.Tensor:
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
        weights = compute_eq_weights(
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
    """Flatten the grid tensor to a tensor of shape
    (batch_size, num_grid_points, dim).

    Arguments:
        grid: Tensor of shape (batch_size, n1, n2, ..., ndim, dim)

    Returns:
        Tensor of shape (batch_size, num_grid_points, dim)
    """

    return tf.reshape(grid, shape=(grid.shape[0], -1, grid.shape[-1]))


def compute_eq_weights(
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
