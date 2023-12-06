from typing import Tuple
import tensorflow as tf

tfk = tf.keras

CONV = {
    1: tfk.layers.Conv1D,
    2: tfk.layers.Conv2D,
    3: tfk.layers.Conv3D,
}

TRANSPOSE_CONV = {
    1: tfk.layers.Conv1DTranspose,
    2: tfk.layers.Conv2DTranspose,
    3: tfk.layers.Conv3DTranspose,
}


class UNetBlock(tf.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        subnet_channels: Tuple[int],
        stride: int,
        kernel_size: int,
        dim: int,
        seed: int,
        name="unet_block",
        **kwargs,
    ):
        assert dim in [
            1,
            2,
            3,
        ], f"UNet dim must be in [1, 2, 3], found {dim=}."

        super().__init__(name=name, **kwargs)

        if len(subnet_channels) == 0:
            self.subnet = tf.identity
            self.conv_down = None
            self.conv_up = None

        else:
            seed = seed + 1
            self.subnet = UNetBlock(
                in_channels=subnet_channels[0],
                subnet_channels=subnet_channels[1:],
                kernel_size=kernel_size,
                stride=stride,
                dim=dim,
                seed=seed,
            )

            seed = seed + 1
            self.conv_down = CONV[dim](
                filters=subnet_channels[0],
                strides=stride,
                kernel_size=kernel_size,
                padding="same",
                activation=None,
                use_bias=True,
                kernel_initializer=tfk.initializers.GlorotUniform(seed=seed),
            )
            self.norm_down = tfk.layers.BatchNormalization()

            up_channels = (
                subnet_channels[0]
                if len(subnet_channels) == 1
                else 2 * subnet_channels[0]
            )

            seed = seed + 1
            self.conv_up = TRANSPOSE_CONV[dim](
                filters=up_channels,
                strides=stride,
                padding="same",
                kernel_size=kernel_size,
                activation=None,
                use_bias=True,
                kernel_initializer=tfk.initializers.GlorotUniform(seed=seed),
            )
            self.norm_up = tfk.layers.BatchNormalization()

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        skip = x

        # Apply down convolution
        if self.conv_down is not None:
            x = tf.nn.relu(
                self.norm_down(
                    self.conv_down(x),
                    training=training,
                )
            )

        # Apply subnet recursively
        x = self.subnet(x)

        # Apply up convolution and concatenate with skip connection
        if self.conv_up is not None:
            x = tf.nn.relu(
                self.norm_up(
                    self.conv_up(x),
                    training=training,
                )
            )
            x = tf.concat([x, skip], axis=-1)

        return x


class UNet(tf.Module):
    def __init__(
        self,
        *,
        first_channels: int,
        last_channels: int,
        kernel_size: int,
        num_channels: Tuple[int],
        stride: int,
        dim: int,
        seed: int,
        name: str = "unet",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        assert len(num_channels) > 0, "UNet must have at least one layer."

        seed = seed + 1
        self.first = CONV[dim](
            filters=first_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding="same",
            activation=None,
            use_bias=True,
            kernel_initializer=tfk.initializers.GlorotUniform(seed=seed),
        )

        seed = seed + 1
        self.last = TRANSPOSE_CONV[dim](
            filters=last_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding="same",
            activation=None,
            use_bias=True,
            kernel_initializer=tfk.initializers.GlorotUniform(seed=seed),
        )

        seed = seed + 1
        self.unet = UNetBlock(
            in_channels=first_channels,
            subnet_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            dim=dim,
            seed=seed,
        )

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.last(
            self.unet(
                tf.nn.relu(self.first(x)),
                training=training,
            )
        )
