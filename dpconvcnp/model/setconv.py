from typing import Tuple, Optional, List

import tensorflow as tf
from check_shape import check_shape

from dpconvcnp.random import zero_mean_mvn_on_grid_from_chol
from dpconvcnp.random import Seed
from dpconvcnp.utils import f64, f32, cast, logit, to_tensor, expand_last_dims
from dpconvcnp.model.privacy_accounting import (
    sens_per_sigma as dp_sens_per_sigma,
)

tfkl = tf.keras.layers


class DPSetConvEncoder(tf.Module):
    def __init__(
        self,
        *,
        seed: int,
        points_per_unit: int,
        lengthscale_init: float,
        y_bound_init: float,
        w_noise_init: float,
        margin: float,
        lengthscale_trainable: bool,
        y_bound_trainable: bool,
        w_noise_trainable: bool,
        amortize_y_bound: bool,
        amortize_w_noise: bool,
        num_mlp_hidden_units: int,
        dim: int,
        n_norm_factor: float = 512.0,
        xmin: Optional[List[float]] = None,
        xmax: Optional[List[float]] = None,
        dtype: tf.DType = tf.float32,
        name="dp_set_conv",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.points_per_unit = points_per_unit

        lengthscale_init = to_tensor(dim * [lengthscale_init], dtype=dtype)
        self.log_lengthscales = tf.Variable(
            initial_value=tf.math.log(lengthscale_init),
            trainable=lengthscale_trainable,
            dtype=dtype,
        )

        self.amortize_y_bound = amortize_y_bound
        self.amortize_w_noise = amortize_w_noise

        if self.amortize_y_bound:
            self._log_y_bound = MLP(
                seed=seed,
                num_hidden_units=num_mlp_hidden_units,
                num_output_units=1,
                dtype=dtype,
            )
            seed = seed + 1

        else:
            self._log_y_bound = tf.Variable(
                initial_value=tf.math.log(y_bound_init),
                trainable=y_bound_trainable,
                dtype=dtype,
            )

        if self.amortize_w_noise:
            self._logit_w_noise = MLP(
                seed=seed,
                num_hidden_units=num_mlp_hidden_units,
                num_output_units=1,
                dtype=dtype,
            )

        else:
            self._logit_w_noise = tf.Variable(
                initial_value=logit(w_noise_init),
                trainable=w_noise_trainable,
                dtype=dtype,
            )

        self.margin = margin

        self.xmin = to_tensor(xmin, dtype=dtype) if xmin is not None else None
        self.xmax = to_tensor(xmax, dtype=dtype) if xmax is not None else None

        self.n_norm_factor = n_norm_factor

    def log_y_bound(self, sens_num_ctx: tf.Tensor) -> tf.Tensor:
        if self.amortize_y_bound:
            return self._log_y_bound(sens_num_ctx)

        else:
            return self._log_y_bound[None, None]

    def logit_w_noise(self, sens_num_ctx: tf.Tensor) -> tf.Tensor:
        if self.amortize_w_noise:
            return self._logit_w_noise(sens_num_ctx)

        else:
            return self._logit_w_noise[None, None]

    @property
    def lengthscales(self) -> tf.Tensor:
        return tf.exp(self.log_lengthscales)

    def y_bound(self, sens_num_ctx: tf.Tensor) -> tf.Tensor:
        y_bound = tf.exp(self.log_y_bound(sens_num_ctx=sens_num_ctx))
        return y_bound + 1e-2

    def w_noise(self, sens_num_ctx: tf.Tensor) -> tf.Tensor:
        w_noise = tf.nn.sigmoid(
            self.logit_w_noise(sens_num_ctx=sens_num_ctx)
        )
        return (1 - 2e-2) * w_noise + 1e-2

    def data_sigma(
        self,
        sens_per_sigma: tf.Tensor,
        num_ctx: tf.Tensor,
    ) -> tf.Tensor:
        sens_num_ctx = tf.stack(
            [sens_per_sigma, num_ctx / self.n_norm_factor],
            axis=-1,
        )
        y_bound = self.y_bound(sens_num_ctx=sens_num_ctx)
        w_noise = self.w_noise(sens_num_ctx=sens_num_ctx)

        return 2.0 * y_bound[:, 0] / (sens_per_sigma * w_noise[:, 0] ** 0.5)

    def density_sigma(
        self,
        sens_per_sigma: tf.Tensor,
        num_ctx: tf.Tensor,
    ) -> tf.Tensor:
        sens_num_ctx = tf.stack(
            [sens_per_sigma, num_ctx / self.n_norm_factor],
            axis=-1,
        )
        w_noise = self.w_noise(sens_num_ctx=sens_num_ctx)

        return 2**0.5 / (sens_per_sigma * (1 - w_noise[:, 0]) ** 0.5)

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

        # Compute sensitivity per sigma
        sens_per_sigma = dp_sens_per_sigma(epsilon=epsilon, delta=delta)

        # Clip context outputs and concatenate tensor of ones
        y_ctx = self.clip_y(
            y_ctx=y_ctx,
            sens_per_sigma=sens_per_sigma,
        )  # shape (batch_size, num_ctx, 1)

        y_ctx = tf.concat(
            [y_ctx, tf.ones_like(y_ctx)],
            axis=-1,
        )  # shape (batch_size, num_ctx, Dy+1)

        # Build dimension wise grids
        x_grid, x_dimension_wise_grids = (
            make_adaptive_grids(
                x=tf.concat([x_ctx, x_trg], axis=1),
                points_per_unit=self.points_per_unit,
                margin=self.margin,
            )
            if self.xmin is None or self.xmax is None
            else make_grids(
                xmin=tf.tile(self.xmin[None, :], [tf.shape(x_ctx)[0], 1]),
                xmax=tf.tile(self.xmax[None, :], [tf.shape(x_ctx)[0], 1]),
                points_per_unit=self.points_per_unit,
                margin=self.margin,
            )
        )  # list of tensors of shape (batch_size, 2*N[d]+1)

        x_grid_flat = flatten_grid(
            x_grid
        )  # shape (batch_size, num_grid_points, Dx)

        # Compute matrix of weights between context points and grid points
        weights = compute_eq_weights(
            x1=x_grid_flat,
            x2=x_ctx,
            lengthscales=self.lengthscales,
        )  # shape (batch_size, num_ctx, num_grid_points)

        # Multiply context outputs by weights
        z_grid_flat = tf.matmul(
            weights,
            y_ctx,
        )  # shape (batch_size, num_grid_points, 2)

        # Reshape grid
        z_grid = tf.reshape(
            z_grid_flat,
            shape=tf.concat(
                [tf.shape(x_grid)[:-1], tf.shape(z_grid_flat)[-1:]],
                axis=0,
            ),
        )  # shape (batch_size, n1, ..., ndim, 2)

        # Sample noise and add it to the data and density channels
        num_ctx = tf.reduce_sum(tf.ones_like(y_ctx[0, :, 0]))
        seed, noise, noise_std = self.sample_noise(
            seed=seed,
            x_dimension_wise_grids=x_dimension_wise_grids,
            sens_per_sigma=sens_per_sigma,
            num_ctx=num_ctx,
        )

        # Add noise to data and density channels
        z_grid = z_grid + noise

        # Concatenate noise standard deviation to grid
        num_ctx = tf.ones_like(z_grid[..., :1]) * num_ctx / self.n_norm_factor
        z_grid = tf.concat([z_grid, noise_std, num_ctx], axis=-1)

        return seed, x_grid, z_grid

    def clip_y(self, sens_per_sigma: tf.Tensor, y_ctx: tf.Tensor) -> tf.Tensor:
        """Clip the context outputs to be within the range [-y_bound, y_bound].

        Arguments:
            y_ctx: Tensor of shape (batch_size, num_ctx, dim) containing the
                context outputs.

        Returns:
            Tensor of shape (batch_size, num_ctx, dim) containing the clipped
                context outputs.
        """

        y_bound = self.y_bound(sens_per_sigma=sens_per_sigma)[:, :, None]
        return tf.clip_by_value(y_ctx, -y_bound, y_bound)

    def sample_noise(
        self,
        seed: Seed,
        x_dimension_wise_grids: List[tf.Tensor],
        sens_per_sigma: tf.Tensor,
        num_ctx: tf.Tensor,
    ) -> Tuple[Seed, tf.Tensor, tf.Tensor]:
        """Sample noise for the density and data channels, returning the new
        seed, the noise tensor and the noise standard deviation tensor.

        Arguments:
            x_grid: Tensor of shape (batch_size, num_grid_points, Dx)
            sens_per_sigma: Tensor of shape (batch_size,), sensitivity per sigma

        Returns:
            Tensor of shape (batch_size, num_grid_points, 2)
        """

        # Check sensitivity per sigma does not contain NaNs or infs
        tf.debugging.assert_all_finite(
            sens_per_sigma,
            message="sens_per_sigma contains NaNs or inf.",
        )

        # Get input data type
        in_dtype = x_dimension_wise_grids[0].dtype

        kxx_dimension_wise = [
            compute_eq_weights(
                x1=cast(x_dimension_wise_grids[i][:, :, None], dtype=f64),
                x2=cast(x_dimension_wise_grids[i][:, :, None], dtype=f64),
                lengthscales=cast(self.lengthscales[i : i + 1], dtype=f64),
            )
            for i in range(len(x_dimension_wise_grids))
        ]

        kxx_chol_dimension_wise = [
            tf.linalg.cholesky(
                kxx + 1e-6 * tf.eye(tf.shape(kxx)[-1], dtype=kxx.dtype)
            )
            for kxx in kxx_dimension_wise
        ]

        # Draw noise samples for the density and data channels
        seed, data_noise = zero_mean_mvn_on_grid_from_chol(
            seed=seed,
            cov_chols=kxx_chol_dimension_wise[::-1],
        )  # shape (batch_size, n1, ..., nd)

        seed, density_noise = zero_mean_mvn_on_grid_from_chol(
            seed=seed,
            cov_chols=kxx_chol_dimension_wise[::-1],
        )  # shape (batch_size, n1, ..., nd)

        # Convert back to input data types
        data_noise = cast(data_noise, dtype=in_dtype)
        density_noise = cast(density_noise, dtype=in_dtype)

        # Get stacked tensors with sensitivity per sigma and number of context
        # points, with the latter scaled down by the normalization factor
        num_ctx = cast(num_ctx[:, None], f32)

        # Compute and expand data_sigma and density_sigma for broadcasting
        data_sigma = expand_last_dims(
            self.data_sigma(sens_per_sigma=sens_per_sigma, num_ctx=num_ctx),
            ndims=len(tf.shape(data_noise)) - 1,
        )  # shape (batch_size, 1, ..., 1)

        density_sigma = expand_last_dims(
            self.density_sigma(sens_per_sigma=sens_per_sigma, num_ctx=num_ctx),
            ndims=len(tf.shape(density_noise)) - 1,
        )  # shape (batch_size, 1, ..., 1)

        # Multiply noise by standard deviations
        noise = tf.stack(
            [
                data_noise * data_sigma,
                density_noise * density_sigma,
            ],
            axis=-1,
        )  # shape (batch_size, n1, ..., nd, 2)

        noise_std = tf.stack(
            [
                tf.ones_like(data_noise) * data_sigma,
                tf.ones_like(density_noise) * density_sigma,
            ],
            axis=-1,
        )  # shape (batch_size, n1, ..., nd, 2)

        return seed, noise, noise_std


class SetConvDecoder(tf.Module):
    def __init__(
        self,
        *,
        lengthscale_init: float,
        scaling_factor: float,
        dim: int,
        dtype: tf.DType = tf.float32,
        lengthscale_trainable: bool = True,
    ):
        super().__init__()

        lengthscale_init = to_tensor(dim * [lengthscale_init], dtype=dtype)
        self.log_lengthscales = tf.Variable(
            initial_value=tf.math.log(lengthscale_init),
            trainable=lengthscale_trainable,
        )

        self.scaling_factor = scaling_factor

    @property
    def lengthscales(self) -> tf.Tensor:
        return tf.exp(self.log_lengthscales)

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
        x_grid = flatten_grid(
            x_grid
        )  # shape (batch_size, num_grid_points, Dx)
        z_grid = flatten_grid(
            z_grid
        )  # shape (batch_size, num_grid_points, Dz)

        # Compute weights
        weights = compute_eq_weights(
            x1=x_trg,
            x2=x_grid,
            lengthscales=self.lengthscales,
        )  # shape (batch_size, num_trg, num_grid_points)

        z_grid = tf.matmul(weights, z_grid) / self.scaling_factor

        return z_grid  # shape (batch_size, num_trg, Dz)


class MLP(tf.Module):
    def __init__(
        self,
        *,
        seed: int,
        num_hidden_units: int,
        num_output_units: int,
        activation: str = "relu",
        scaling_factor: float = 10.0,
        dtype: tf.DType = tf.float32,
        name: str = "mlp",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dense1 = tfkl.Dense(
            units=num_hidden_units,
            activation=activation,
            kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
            dtype=dtype,
        )
        seed = seed + 1

        self.dense2 = tfkl.Dense(
            units=num_hidden_units,
            activation=activation,
            kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
            dtype=dtype,
        )
        seed = seed + 1

        self.dense3 = tfkl.Dense(
            units=num_output_units,
            activation=None,
            kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
            dtype=dtype,
        )

        self.scaling_factor = scaling_factor

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return self.dense3(self.dense2(self.dense1(x))) / self.scaling_factor


def make_adaptive_grids(
    x: tf.Tensor,
    points_per_unit: int,
    margin: float,
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """Create grids

    Arguments:
        x: Tensor of shape (batch_size, num_points, dim) containing the
            points.
        points_per_unit: Number of points per unit length in each dimension.
        margin: Margin around the points in `x`.

    Returns:
        Tensor of shape (batch_size, n1, n2, ..., ndim, dim)
    """

    # Compute the lower and upper corners of the box containing the points
    xmin = tf.reduce_min(x, axis=1)
    xmax = tf.reduce_max(x, axis=1)

    return make_grids(
        xmin=xmin,
        xmax=xmax,
        points_per_unit=points_per_unit,
        margin=margin,
    )


def make_grids(
    xmin: tf.Tensor,
    xmax: tf.Tensor,
    points_per_unit: int,
    margin: float,
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """Create grids

    Arguments:
        xmin: Tensor of shape (batch_size, dim) containing the lower
            corner of the box.
        xmax: Tensor of shape (batch_size, dim) containing the upper
            corner of the box.
        points_per_unit: Number of points per unit length in each dimension.
        margin: Margin around the box.

    Returns:
        Tensor of shape (batch_size, n1, n2, ..., ndim, dim)
    """

    # Get grid dimension
    dim = xmin.shape[-1]

    # Compute half the number of points in each dimension
    N = tf.math.ceil(
        (0.5 * (xmax - xmin) + margin) * points_per_unit
    )  # shape (batch_size, dim)

    # Take the maximum over the batch, in order to use the same number of
    # points across all tasks in the batch, to enable tensor batching
    N = tf.reduce_max(N, axis=0)  # shape (dim,)
    N = 2 ** tf.math.ceil(tf.math.log(N) / tf.math.log(2.0))  # shape (dim,)

    # Compute midpoints of each dimension, multiply integer grid by the grid
    # spacing and add midpoint to obtain dimension-wise grids
    x_mid = 0.5 * (xmin + xmax)  # shape (batch_size, dim)

    # Set up list of dimension-wise grids
    dimension_wise_grids = [
        x_mid[:, i : i + 1]
        + tf.range(-N[i], N[i], dtype=xmin.dtype)[None, :] / points_per_unit
        for i in range(dim)
    ]  # list of tensors with shapes (batch_size, 2*N[d]+1)

    # Compute multi-dimensional grid
    grid = tf.stack(
        tf.meshgrid(
            *[tf.range(-N[-i], N[-i], dtype=xmin.dtype) for i in range(dim)]
        ),
        axis=-1,
    )  # shape (n1, n2, ..., ndim, dim)

    for _ in range(dim):
        x_mid = tf.expand_dims(x_mid, axis=1)

    # Multiply integer grid by the grid spacing and add midpoint
    grid = x_mid + grid[None, ...] / points_per_unit

    return grid, dimension_wise_grids


def flatten_grid(grid: tf.Tensor) -> tf.Tensor:
    """Flatten the grid tensor to a tensor of shape
    (batch_size, num_grid_points, dim).

    Arguments:
        grid: Tensor of shape (batch_size, n1, n2, ..., ndim, dim)

    Returns:
        Tensor of shape (batch_size, num_grid_points, dim)
    """
    return tf.reshape(grid, shape=(tf.shape(grid)[0], -1, tf.shape(grid)[-1]))


def compute_eq_weights(
    x1: tf.Tensor,
    x2: tf.Tensor,
    lengthscales: tf.Tensor,
) -> tf.Tensor:
    """Compute the weights for the SetConv layer, mapping from `x1` to `x2`.

    Arguments:
        x1: Tensor of shape (batch_size, num_x1, dim)
        x2: Tensor of shape (batch_size, num_x2, dim)
        lengthscales: Tensor of shape (dim,)

    Returns:
        Tensor of shape (batch_size, num_x1, num_x2)
    """

    # Expand dimensions for broadcasting
    x1 = x1[:, :, None, :]
    x2 = x2[:, None, :, :]
    lengthscales = lengthscales[None, None, None, :]

    # Compute pairwise distances between x1 and x2
    dist2 = tf.reduce_sum(
        ((x1 - x2) / lengthscales) ** 2.0,
        axis=-1,
    )  # shape (batch_size, num_x1, num_x2)

    # Compute weights
    weights = tf.exp(-0.5 * dist2)  # shape (batch_size, num_x1, num_x2)

    return weights
