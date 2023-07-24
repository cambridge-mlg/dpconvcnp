from typing import Tuple
import tensorflow as tf
import tensorflow_probability as tfp

from dpconvcnp.data.data import Batch
from dpconvcnp.random import Seed
from dpconvcnp.model.setconv import DPSetConvEncoder, SetConvDecoder

tfd = tfp.distributions


class DPConvCNP(tf.keras.Model):

    def __init__(
        self,
        conv_net: tf.Module,
        dpsetconv_encoder: DPSetConvEncoder,
        setconv_decoder: SetConvDecoder,
        name: str = "dpconvcp",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dpsetconv_encoder = dpsetconv_encoder
        self.setconv_decoder = setconv_decoder
        self.conv_net = conv_net
        

    def __call__(
        self,
        seed: Seed,
        x_ctx: tf.Tensor,
        y_ctx: tf.Tensor,
        x_trg: tf.Tensor,
        epsilon: tf.Tensor,
        delta: tf.Tensor,
    ) -> Tuple[Seed, tf.Tensor, tf.Tensor]:
        
        seed, x_grid, z_grid = self.dpsetconv_encoder(
            seed=seed,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_trg=x_trg,
            epsilon=epsilon,
            delta=delta,
        )

        z_grid = self.conv_net(z_grid)

        z_trg = self.setconv_decoder(
            x_grid=x_grid,
            z_grid=z_grid,
            x_trg=x_trg,
        )

        mean = z_trg[..., :1]
        std = tf.math.softplus(z_trg[..., 1:])**0.5

        return seed, mean, std


    @tf.function(experimental_relax_shapes=True) 
    def loss(
        self,
        seed: Seed,
        x_ctx: tf.Tensor,
        y_ctx: tf.Tensor,
        x_trg: tf.Tensor,
        y_trg: tf.Tensor,
        epsilon: tf.Tensor,
        delta: tf.Tensor,
    ) -> Tuple[Seed, tf.Tensor, tf.Tensor, tf.Tensor]:

        seed, mean, std = self.__call__(
            seed=seed,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_trg=x_trg,
            epsilon=epsilon,
            delta=delta,
        )

        log_prob = tfd.Normal(loc=mean, scale=std).log_prob(y_trg)
        loss = - tf.reduce_sum(log_prob, axis=[1, 2])

        return seed, loss, mean, std
