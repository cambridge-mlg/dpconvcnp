from typing import Any, Dict
import tensorflow as tf
import tensorflow_probability as tfp

from dpconvcnp.data.data import Batch
from dpconvcnp.random import Seed
from dpconvcnp.model.setconv import DPSetConvEncoder, SetConvDecoder

tfd = tfp.distributions


class DPConvCNP(tf.Module):

    def __init__(
        self,
        conv_arch: Dict[str, Any],
        points_per_unit: int,
        lenghtscale_init: float,
        y_bound_init: float,
        w_noise_init: float,
        encoder_lengthscale_trainable: bool,
        y_bound_trainable: bool,
        w_noise_trainable: bool,
        decoder_lengthscale_trainable: bool = True,
        dtype: tf.DType = tf.float32,
        name: str = "dpconvcp",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dpsetconv_encoder = DPSetConvEncoder(
            points_per_unit=points_per_unit,
            lenghtscale_init=lenghtscale_init,
            y_bound_init=y_bound_init,
            w_noise_init=w_noise_init,
            lengthscale_trainable=encoder_lengthscale_trainable,
            y_bound_trainable=y_bound_trainable,
            w_noise_trainable=w_noise_trainable,
            dtype=dtype,
        )

        self.setconv_decoder = SetConvDecoder(
            points_per_unit=points_per_unit,
            trainable=decoder_lengthscale_trainable,
        )

        self.conv_arch = self.build_conv_arch(conv_arch=conv_arch)

    def build_conv_arch(self, conv_arch: Dict[str, Any]):
        pass

    def __call__(self, seed: Seed, batch: Batch, training: bool = False):
        
        seed, x_grid, z_grid = self.dpsetconv_encoder(
            seed=seed,
            x_ctx=batch.x_ctx,
            y_ctx=batch.y_ctx,
            x_trg=batch.x_trg,
            epsilon=batch.epsilon,
            delta=batch.delta,
        )

        z_grid = self.conv_arch(z, training=training)

        z_trg = self.setconv_decoder(
            x_grid=x_grid,
            z_grid=z_grid,
            x_trg=batch.x_trg,
        )

        assert z_trg.shape[-1] == 2

        mean = z_trg[..., 0]
        std = tf.math.softplus(z_trg[..., 1])

        return seed, mean, std

    def loss(self, seed: Seed, batch: Batch):

        seed, mean, std = self.__call__(seed=seed, batch=batch, training=True)

        log_prob = tfd.Normal(loc=mean, scale=std).log_prob(batch.y_trg)

        return tf.reduce_sum(log_prob, axis=[1, 2])