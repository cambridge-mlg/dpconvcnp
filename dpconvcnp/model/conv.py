from typing import Tuple
import tensorflow as tf

class UNet(tf.Module):

    def __init__(
            self,
            num_channels: Tuple[int],
            strides: Tuple[int],
            dim_input: int,
            dim_output: int,
            activation: str = "relu",
            name="unet",
            **kwargs,
        ):

        assert len(num_channels) == len(strides) or len(strides) == 1, (
            f"UNet num_channels and strides must have the same length, or "
            f"strides must have length 1, found {len(num_channels)=} and "
            f"{len(strides)=}."
        )

        super().__init__(name=name, **kwargs)

        self.num_channels = num_channels
        self.strides = strides
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.activation = activation

        self.convs = []
        self.transposed_convs = []

        with self.name_scope:
            pass


    def __call__(self, z: tf.Tensor):
        pass