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


class UNet(tf.Module):

    def __init__(
            self,
            *,
            first_channels: int,
            last_channels: int,
            kernel_size: int,
            num_channels: Tuple[int],
            strides: Tuple[int],
            dim: int,
            seed: int,
            activation: str = "relu",
            name="unet",
            **kwargs,
        ):
        """Constructs a UNet-based convolutional architecture, consisting
        of a first convolutional layer, followed by a UNet style architecture
        with skip connections, and a final convolutional layer.

        Arguments:
            first_channels: Number of channels in the first convolutional layer.
            last_channels: Number of channels in the last convolutional layer.
            kernel_size: Size of the convolutional kernels.
            num_channels: Number of channels in each UNet layer.
            strides: Strides in each UNet layer.
            dim: Dimensionality of the input data.
            seed: Random seed.
            activation: Activation function to use.
            name: Name of the module.
            **kwargs: Additional keyword arguments.
        """

        assert len(num_channels) == len(strides), (
            f"UNet num_channels and strides must have the same length, found "
            f"{len(num_channels)=} and {len(strides)=}."
        )

        assert dim in [1, 2, 3], (
            f"UNet dim must be in [1, 2, 3], found {dim=}."
        )

        super().__init__(name=name, **kwargs)

        self.convs = []
        self.transposed_convs = []

        shared_kwargs = lambda f, k, s, a, seed: {
            "filters": f,
            "kernel_size": k,
            "strides": s,
            "activation": a,
            "padding": "same",
            "data_format": "channels_last",
            "kernel_initializer": tfk.initializers.GlorotUniform(seed=seed),
        }

        with self.name_scope:

            # First convolutional layer
            self.first = CONV[dim](
                **shared_kwargs(first_channels, kernel_size, 1, None, seed)
            )
            seed += 1

            # UNet layers
            for i in range(len(num_channels)):

                self.convs.append(
                    CONV[dim](
                        **shared_kwargs(num_channels[i], kernel_size, 2, activation, seed)
                    )
                )
                seed += 1

                self.transposed_convs.append(
                    TRANSPOSE_CONV[dim](
                        **shared_kwargs(num_channels[-i-1], kernel_size, 2, activation, seed)
                    )
                )
                seed += 1

            # Last convolutional layer
            self.last = TRANSPOSE_CONV[dim](
                **shared_kwargs(last_channels, kernel_size, 1, None, seed)
            )

    def __call__(self, z: tf.Tensor):

        z = self.first(z)
        skips = []

        for conv in self.convs:
            skips.append(z)
            z = conv(z)

        for conv, skip in zip(self.transposed_convs, skips[::-1]):
            z = conv(z)
            z = tf.concat([z, skip], axis=-1)
            
        z = self.last(z)

        return z
