from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from dpconvcnp.data.data import Batch, DataGenerator
from dpconvcnp.random import Seed, randperm, randint, to_tensor
from dpconvcnp.utils import f32, i32


def _scale(x: tf.Tensor) -> tf.Tensor:
    return (
        2.0 * (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
        - 1.0
    )
    
def _normalise(x: tf.Tensor) -> tf.Tensor:
    return (
        (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x))
    )

def _select_batch_of_tasks(tensor: tf.Tensor, idx: tf.Tensor) -> tf.Tensor:
    shape = idx.shape
    idx = tf.reshape(idx, (-1,))
    return tf.reshape(tf.gather(tensor, idx), shape)


class HowellGenerator(DataGenerator):
    field_names: Tuple[str] = ["age", "height", "weight"]

    def __init__(
        self,
        *,
        min_num_ctx: int,
        max_num_ctx: int,
        x_name: str,
        y_name: str,
        reset_seed_at_epoch_end: bool = False,
        **kwargs,
    ):
        assert x_name in self.field_names, f"Invalid x_name {x_name}"
        assert y_name in self.field_names, f"Invalid y_name {y_name}"
        assert x_name != y_name, f"x_name and y_name must be different"

        super().__init__(**kwargs)

        self.base_seed = self.seed
        self.reset_seed_at_epoch_end = reset_seed_at_epoch_end

        # Set dataloader parameters
        self.min_num_ctx = to_tensor(min_num_ctx, i32)
        self.max_num_ctx = to_tensor(max_num_ctx, i32)

        # Load data fields
        self.data = self.load_full_data()

        # Set task parameters
        self.x_name = x_name
        self.y_name = y_name
        self.x = _scale(to_tensor(self.data[x_name], f32))
        self.y = 1.5 * _normalise(to_tensor(self.data[y_name], f32))
        self.num_data = to_tensor(self.x.shape[0], i32)

        assert (
            max_num_ctx < self.num_data
        ), "max_num_context must be less than the number of data points"

    def load_full_data(self) -> Dict[str, np.array]:
        data = {}
        dataset = tfds.load("howell", split="train")
        for datapoint in dataset:
            for k in self.field_names:
                data[k] = data.get(k, [])
                data[k].append(datapoint[k][None])

        for k, v in data.items():
            data[k] = tf.concat(v, axis=0)

        return data

    def __iter__(self):
        """Reset epoch counter and seed and return self."""
        self.seed = (
            self.base_seed if self.reset_seed_at_epoch_end else self.seed
        )
        return super().__iter__()

    def generate_data(self, seed: Seed) -> Tuple[Seed, Batch]:
        # Sample number of context points
        seed, num_ctx = randint(
            seed=seed,
            shape=(),
            minval=self.min_num_ctx,
            maxval=self.max_num_ctx,
        )

        # Set up range of indices for each task
        seed, idx = randperm(
            seed=seed,
            shape=(self.batch_size,),
            maxval=self.num_data - 1,
        )

        # Split indices into context and target
        ctx_idx = idx[:, :num_ctx]
        trg_idx = idx[:, num_ctx:]

        # Get tasks for batch
        x_ctx = _select_batch_of_tasks(self.x, ctx_idx)[:, :, None]
        y_ctx = _select_batch_of_tasks(self.y, ctx_idx)[:, :, None]
        x_trg = _select_batch_of_tasks(self.x, trg_idx)[:, :, None]
        y_trg = _select_batch_of_tasks(self.y, trg_idx)[:, :, None]

        return seed, Batch(
            x=self.x,
            y=self.y,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_trg=x_trg,
            y_trg=y_trg,
        )
