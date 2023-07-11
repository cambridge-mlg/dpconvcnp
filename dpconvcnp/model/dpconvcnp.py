from typing import Any, Dict, Tuple
import tensorflow as tf
import tensorflow_probability as tfp

from dpconvcnp.data.data import Batch
from dpconvcnp.random import Seed
from dpconvcnp.model.setconv import DPSetConvEncoder, SetConvDecoder
from dpconvcnp.model.conv import UNet
from dpconvcnp.utils import to_tensor

tfd = tfp.distributions


CONV_ARCHITECTURES = {
    "unet": UNet,
}


class DPConvCNP(tf.Module):

    def __init__(
        self,
        points_per_unit: int,
        margin: float,
        lenghtscale_init: float,
        y_bound_init: float,
        w_noise_init: float,
        encoder_lengthscale_trainable: bool,
        y_bound_trainable: bool,
        w_noise_trainable: bool,
        architcture: str,
        architcture_kwargs: Dict[str, Any],
        decoder_lengthscale_trainable: bool = True,
        dtype: tf.DType = tf.float32,
        name: str = "dpconvcp",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        assert architcture in CONV_ARCHITECTURES, (
            f"conv_arch['name'] must be in {CONV_ARCHITECTURES.keys()}, "
            f"found {architcture=}."
        )

        self.dpsetconv_encoder = DPSetConvEncoder(
            points_per_unit=points_per_unit,
            margin=margin,
            lenghtscale_init=lenghtscale_init,
            y_bound_init=y_bound_init,
            w_noise_init=w_noise_init,
            lengthscale_trainable=encoder_lengthscale_trainable,
            y_bound_trainable=y_bound_trainable,
            w_noise_trainable=w_noise_trainable,
            dtype=dtype,
        )

        self.setconv_decoder = SetConvDecoder(
            lengthscale_init=lenghtscale_init,
            trainable=decoder_lengthscale_trainable,
        )

        self.conv_arch = CONV_ARCHITECTURES[architcture](**architcture_kwargs)

        
    def __call__(
        self,
        seed: Seed,
        x_ctx: tf.Tensor,
        y_ctx: tf.Tensor,
        x_trg: tf.Tensor,
        epsilon: tf.Tensor,
        delta: tf.Tensor,
    ):
        
        seed, x_grid, z_grid = self.dpsetconv_encoder(
            seed=seed,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_trg=x_trg,
            epsilon=epsilon,
            delta=delta,
        )

        z_grid = self.conv_arch(z_grid)

        z_trg = self.setconv_decoder(
            x_grid=x_grid,
            z_grid=z_grid,
            x_trg=x_trg,
        )

        assert z_trg.shape[-1] == 2

        mean = z_trg[..., :1]
        std = tf.math.softplus(z_trg[..., 1:])**0.5

        return seed, mean, std


    def loss(
        self,
        seed: Seed,
        x_ctx: tf.Tensor,
        y_ctx: tf.Tensor,
        x_trg: tf.Tensor,
        y_trg: tf.Tensor,
        epsilon: tf.Tensor,
        delta: tf.Tensor,
    ):

        seed, mean, std = self.__call__(
            seed=seed,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_trg=x_trg,
            epsilon=epsilon,
            delta=delta,
        )

        log_prob = tfd.Normal(loc=mean, scale=std).log_prob(y_trg)

        return seed, - tf.reduce_sum(log_prob, axis=[1, 2])
