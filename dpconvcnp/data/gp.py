from abc import abstractmethod, ABC
from typing import Tuple, Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import numpy as np

from dpconvcnp.data.data import SyntheticGenerator, GroundTruthPredictor
from dpconvcnp.random import randu, zero_mean_mvn
from dpconvcnp.random import Seed
from dpconvcnp.utils import f32, f64, to_tensor, cast
from dpconvcnp.model.privacy_accounting import (
    sens_per_sigma as dp_sens_per_sigma,
)
from dpconvcnp.model.setconv import (
    DPSetConvEncoder,
    flatten_grid,
    compute_eq_weights,
)

tfd = tfp.distributions


KERNEL_TYPES = [
    "eq",
    "matern12",
    "matern32",
    "matern52",
    "noisy_mixture",
    "weakly_periodic",
]


class GPGenerator(SyntheticGenerator, ABC):
    def __init__(
        self,
        *,
        dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim

    def sample_outputs(
        self, seed: Seed, x: tf.Tensor
    ) -> Tuple[Seed, tf.Tensor, Callable]:
        """Sample context and target outputs, given the inputs `x`.

        Arguments:
            seed: Random seed.
            x: Tensor of shape (batch_size, num_ctx + num_trg, dim) containing
                the context and target inputs.

        Returns:
            seed: Random seed generated by splitting.
            y: Tensor of shape (batch_size, num_ctx + num_trg, 1) containing
                the context and target outputs.
        """

        # Set up GP kernel
        seed, kernel, kernel_noiseless, noise_std = self.set_up_kernel(
            seed=seed,
        )
        gt_pred = self.set_up_ground_truth_gp(
            kernel=kernel,
            kernel_noiseless=kernel_noiseless,
            noise_std=noise_std,
        )

        # Set up covariance at input locations
        kxx = kernel(cast(x, f64))

        # Sample from GP with zero mean and covariance kxx
        seed, y = zero_mean_mvn(seed=seed, cov=kxx)
        y = tf.expand_dims(y, axis=-1)

        return seed, cast(y, f32), gt_pred

    @abstractmethod
    def set_up_kernel(self, seed: Seed) -> Tuple[Seed, gpflow.kernels.Kernel]:
        """Set up GP kernel.

        Arguments:
            seed: Random seed.

        Returns:
            seed: Random seed generated by splitting.
            kernel: GP kernel.
        """
        pass

    def set_up_ground_truth_gp(
        self,
        kernel: Callable,
        kernel_noiseless: Callable,
        noise_std: float,
    ) -> Callable:
        """Set up GP kernel.

        Arguments:
            seed: Random seed.

        Returns:
            seed: Random seed generated by splitting.
            kernel: GP kernel.
        """
        return GPGroundTruthPredictor(
            kernel=kernel,
            kernel_noiseless=kernel_noiseless,
            noise_std=noise_std,
        )


class RandomScaleGPGenerator(GPGenerator):
    noisy_mixture_long_lengthscale: float = 1.0
    weakly_periodic_period: float = 0.25

    def __init__(
        self,
        *,
        kernel_type: str,
        min_log10_lengthscale: float,
        max_log10_lengthscale: float,
        min_log10_noise_std: Optional[float] = None,
        max_log10_noise_std: Optional[float] = None,
        noise_std: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if noise_std is not None:
            min_log10_noise_std = np.log10(noise_std)
            max_log10_noise_std = np.log10(noise_std)

        self.kernel_type = kernel_type
        self.min_log10_lengthscale = to_tensor(min_log10_lengthscale, f64)
        self.max_log10_lengthscale = to_tensor(max_log10_lengthscale, f64)
        self.min_log10_noise_std = to_tensor(min_log10_noise_std, f64)
        self.max_log10_noise_std = to_tensor(max_log10_noise_std, f64)

        assert (
            self.kernel_type in KERNEL_TYPES
        ), f"kernel_type must be in {KERNEL_TYPES}, found {self.kernel_type=}."

    def set_up_kernel(self, seed: Seed) -> Tuple[Seed, gpflow.kernels.Kernel]:
        # Sample lengthscale
        seed, log10_lengthscale = randu(
            shape=(),
            seed=seed,
            minval=self.min_log10_lengthscale,
            maxval=self.max_log10_lengthscale,
        )
        lengthscale = 10.0**log10_lengthscale

        # Sample noise_std
        seed, log10_noise_std = randu(
            shape=(),
            seed=seed,
            minval=self.min_log10_noise_std,
            maxval=self.max_log10_noise_std,
        )
        noise_std = 10.0**log10_noise_std

        if self.kernel_type == "eq":
            kernel_noiseless = gpflow.kernels.SquaredExponential(
                lengthscales=lengthscale
            )

        elif self.kernel_type == "matern12":
            kernel_noiseless = gpflow.kernels.Matern12(
                lengthscales=lengthscale
            )

        elif self.kernel_type == "matern32":
            kernel_noiseless = gpflow.kernels.Matern32(
                lengthscales=lengthscale
            )

        elif self.kernel_type == "matern52":
            kernel_noiseless = gpflow.kernels.Matern52(
                lengthscales=lengthscale
            )

        elif self.kernel_type == "noisy_mixture":
            kernel_noiseless = gpflow.kernels.SquaredExponential(
                lengthscales=lengthscale,
            ) + gpflow.kernels.SquaredExponential(
                lengthscales=self.noisy_mixture_long_lengthscale,
            )

        elif self.kernel_type == "weakly_periodic":
            kernel_noiseless = gpflow.kernels.SquaredExponential(
                lengthscales=lengthscale,
            ) * gpflow.kernels.Periodic(
                period=self.weakly_periodic_period,
            )

        kernel = kernel_noiseless + gpflow.kernels.White(
            variance=noise_std**2.0,
        )

        return seed, kernel, kernel_noiseless, noise_std


class GPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        kernel: Callable,
        kernel_noiseless: Callable,
        noise_std: float,
    ):
        self.kernel = kernel
        self.kernel_noiseless = kernel_noiseless
        self.noise_std = noise_std

    def __call__(
        self,
        x_ctx: tf.Tensor,
        y_ctx: tf.Tensor,
        x_trg: tf.Tensor,
        y_trg: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        dtype = x_ctx.dtype

        x_ctx = cast(x_ctx, f64)
        y_ctx = cast(y_ctx, f64)
        x_trg = cast(x_trg, f64)
        num_ctx = x_ctx.shape[1]

        k = self.kernel(tf.concat([x_ctx, x_trg], axis=1))
        kcc = k[:, :num_ctx, :num_ctx]
        kct = k[:, :num_ctx, num_ctx:]
        ktc = k[:, num_ctx:, :num_ctx]
        ktt = k[:, num_ctx:, num_ctx:]

        mean = tf.matmul(ktc, tf.linalg.solve(kcc, y_ctx))[:, :, 0]
        cov = ktt - tf.matmul(ktc, tf.linalg.solve(kcc, kct))
        std = tf.sqrt(tf.linalg.diag_part(cov))

        if y_trg is not None:
            y_trg = cast(y_trg, f64)
            gt_log_lik = tfd.Normal(loc=mean, scale=std).log_prob(
                y_trg[:, :, 0]
            )
            gt_log_lik = tf.reduce_sum(gt_log_lik, axis=1)
            gt_log_lik = cast(gt_log_lik, dtype)

        else:
            gt_log_lik = None

        mean = cast(mean, dtype)[:, :, None]
        std = cast(std, dtype)[:, :, None]

        return mean, std, gt_log_lik


class GPWithPrivateOutputsNonprivateInputs:
    def __init__(
        self,
        seed: Seed,
        points_per_unit: int,
        margin: float,
        dpsetconv_lengthscale: float,
        y_bound: Optional[float] = None,
        w_noise: Optional[float] = None,
        dim: Optional[int] = None,
    ):
        y_bound = 2.0 if y_bound is None else y_bound
        w_noise = 0.5 if w_noise is None else w_noise

        self.dpsetconv = DPSetConvEncoder(
            seed=seed,
            points_per_unit=points_per_unit,
            lengthscale_init=dpsetconv_lengthscale,
            y_bound_init=y_bound,
            w_noise_init=w_noise,
            margin=margin,
            lengthscale_trainable=False,
            y_bound_trainable=False,
            w_noise_trainable=False,
            amortize_y_bound=False,
            amortize_w_noise=False,
            num_mlp_hidden_units=None,
            clip_y_ctx=False,
            dim=dim,
        )

        self.lengthscales = self.dpsetconv.lengthscales


    @tf.function(reduce_retracing=True)
    def __call__(
        self,
        seed: Seed,
        epsilon: tf.Tensor,
        delta: tf.Tensor,
        gen_kernel: tf.Tensor,
        gen_kernel_noiseless: tf.Tensor,
        gen_noise_std: float,
        x_ctx: tf.Tensor,
        y_ctx: tf.Tensor,
        x_trg: tf.Tensor,
        y_trg: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Pass data through dpsetconv
        seed, g, z = self.dpsetconv(seed, x_ctx, y_ctx, x_trg, epsilon, delta)

        # Flatten grids and keep only data channel from z
        g = cast(flatten_grid(g), f64)
        c = cast(flatten_grid(x_ctx), f64)
        t = cast(flatten_grid(x_trg), f64)
        z = cast(flatten_grid(z)[..., :1], f64)

        # Cast to f64
        y_trg = cast(y_trg, f64) if y_trg is not None else None
        lengthscales = cast(self.lengthscales, f64)
        gen_noise_std = cast(gen_noise_std, f64)

        # Get number of context points
        num_ctx = c.shape[1]

        # Compute DP noise for outputs
        sens_per_sigma = dp_sens_per_sigma(epsilon=epsilon, delta=delta)
        data_sigma = self.dpsetconv.data_sigma(
            sens_per_sigma=sens_per_sigma,
            num_ctx=None,
        )
        data_sigma = cast(data_sigma, f64)

        # Compute matrices needed for mean and covariance calculations
        K = gen_kernel_noiseless(tf.concat([c, t], axis=1))
        K_cc = K[:, :num_ctx, :num_ctx]
        K_cc = K_cc + 1e-6 * tf.eye(tf.shape(K_cc)[1], dtype=f64)[None, :, :]
        K_ct = K[:, :num_ctx, num_ctx:]
        K_tc = K[:, num_ctx:, :num_ctx]
        K_tt = K[:, num_ctx:, num_ctx:]
        K_cc_plus_noise = gen_kernel(c)

        K_prime_gg = compute_eq_weights(g, g, lengthscales)
        K_prime_gg = data_sigma[:, None, None] ** 2.0 * K_prime_gg

        Phi_gc = compute_eq_weights(g, c, lengthscales)
        Phi_cg = compute_eq_weights(c, g, lengthscales)

        # Compute covariance matrices
        C_ff = K_cc + 1e-6 * tf.eye(tf.shape(K_cc)[1], dtype=f64)[None, :, :]
        C_hh = tf.matmul(Phi_gc, tf.matmul(K_cc_plus_noise, Phi_cg))
        C_hh = C_hh + K_prime_gg
        C_hh = C_hh + 1e-6 * tf.eye(tf.shape(C_hh)[1], dtype=f64)[None, :, :]
        C_hf = tf.matmul(Phi_gc, K_cc)
        C_fh = tf.matmul(K_cc, Phi_cg)

        m = tf.matmul(C_fh, tf.linalg.solve(C_hh, z))
        C = C_ff - tf.matmul(C_fh, tf.linalg.solve(C_hh, C_hf))
        C = C + 1e-6 * tf.eye(tf.shape(C)[1], dtype=f64)[None, :, :]

        mean = tf.matmul(K_tc, tf.linalg.solve(K_cc, m))
        A = tf.matmul(Phi_gc, K_ct)
        cov = K_tt - tf.matmul(A, tf.linalg.solve(C_hh, A), transpose_a=True)
        std = tf.sqrt(tf.linalg.diag_part(cov) + gen_noise_std**2.0)

        if y_trg is not None:
            log_lik = tfd.Normal(
                loc=mean,
                scale=std[..., None],
            ).log_prob(y_trg)
            log_lik = tf.reduce_sum(log_lik, axis=[1, 2])

        else:
            log_lik = None

        return seed, g, mean, std, log_lik
