from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional

import tensorflow as tf

from dpconvcnp.random import randint, randu
from dpconvcnp.utils import f32, i32, to_tensor

Seed = tf.Tensor


@dataclass
class Batch:

    x: Optional[tf.Tensor] = None
    y: Optional[tf.Tensor] = None

    x_ctx: Optional[tf.Tensor] = None
    y_ctx: Optional[tf.Tensor] = None
    x_trg: Optional[tf.Tensor] = None
    y_trg: Optional[tf.Tensor] = None

    epsilon: Optional[tf.Tensor] = None
    delta: Optional[tf.Tensor] = None

class DataGenerator(ABC):

    def __init__(
            self,
            *,
            seed: Seed,
            samples_per_epoch: int,
            batch_size: int,
            min_epsilon: float,
            max_epsilon: float,
            min_log10_delta: float,
            max_log10_delta: float,
        ):
        """Base data generator, which can be used to derive other data generators,
        such as synthetic generators or real data generators.

        Arguments:
            seed: Random seed for generator.
            samples_per_epoch: Number of samples per epoch.
            batch_size: Batch size.
            min_epsilon: Minimum DP epsilon.
            max_epsilon: Maximum DP epsilon.
            min_log10_delta: Minimum log10 DP delta.
            max_log10_delta: Maximum log10 DP delta.
        """

        # Set random seed for generator
        self.seed = seed

        # Set generator parameters
        self.samples_per_epoch = samples_per_epoch
        self.num_batches = samples_per_epoch // batch_size + 1
        self.batch_size = batch_size

        # Set DP parameters
        self.epsilon_range = to_tensor([min_epsilon, max_epsilon], f32)
        self.log10_delta_range = to_tensor([min_log10_delta, max_log10_delta], f32)

        # Set epoch counter
        self.i = 0

    def __iter__(self):
        """Reset epoch counter and return self."""
        self.i = 0
        return self

    def __next__(self) -> Batch:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """
        
        if self.i >= self.num_batches:
            raise StopIteration
        
        else:
            self.i += 1
            self.seed, batch = self.generate_batch(seed=self.seed)
            return batch
        
    @abstractmethod
    def generate_data(self, seed: Seed) -> Tuple[Seed, Batch]:
        """Generate batch of data using the random seed `seed`.

        Arguments:
            seed: Random seed for batch.
        
        Returns:
            seed: Random seed for next batch.
            batch: Tuple of tensors containing the context and target data.
        """
        pass

    def generate_batch(self, seed: Seed) -> Tuple[Seed, Batch]:
        """Generate batch of data using the random seed `seed`.

        Arguments:
            seed: Random seed for batch.
        
        Returns:
            seed: Random seed for next batch.
            batch: Tuple of tensors containing the context and target data,
                as well as the DP epsilon and delta.
        """
        
        # Generate batch, then add in epsilon and delta
        seed, batch = self.generate_data(seed=seed)
        seed, batch.epsilon, batch.delta = self.sample_epsilon_delta(seed=seed)

        return seed, batch

    def sample_epsilon_delta(self, seed: Seed) -> Tuple[Seed, tf.Tensor, tf.Tensor]:
        """Sample epsilon and delta for each task in the batch.
        
        Arguments:
            seed: Random seed.
            
        Returns:
            seed: Random seed generated by splitting.
            epsilon: Tensor of shape (batch_size,) containing the DP epsilon.
            delta: Tensor of shape (batch_size,) containing the DP delta.
        """

        # Sample epsilon
        seed, epsilon = randu(
            shape=(self.batch_size,),
            seed=seed,
            minval=self.epsilon_range[0],
            maxval=self.epsilon_range[1],
        )

        # Sample log10_delta and raise to power of 10
        seed, log10_delta = randu(
            shape=(self.batch_size,),
            seed=seed,
            minval=self.log10_delta_range[0],
            maxval=self.log10_delta_range[1],
        )
        delta = tf.pow(10.0, log10_delta)

        return seed, epsilon, delta


class SyntheticGenerator(DataGenerator, ABC):
    
    def __init__(
        self,
        *,
        min_num_ctx: int,
        max_num_ctx: int,
        min_num_trg: int,
        max_num_trg: int,
        context_range: tf.Tensor,
        target_range: tf.Tensor,
        **kwargs,
    ):

        super().__init__(**kwargs)
        
        # Set synthetic generator parameters
        self.min_num_ctx = to_tensor(min_num_ctx, i32)
        self.max_num_ctx = to_tensor(max_num_ctx, i32)
        self.min_num_trg = to_tensor(min_num_trg, i32)
        self.max_num_trg = to_tensor(max_num_trg, i32)

        self.context_range = to_tensor(context_range, f32)
        self.target_range = to_tensor(target_range, f32)

    def generate_data(self, seed: Seed) -> Tuple[Seed, Batch]:
        """Generate batch of data using the random seed `seed`.

        Arguments:
            seed: Random seed for batch.
        
        Returns:
            seed: Random seed for next batch.
            batch: Tuple of tensors containing the context and target data.
        """
        
        # Sample number of context and target points
        seed, num_ctx, num_trg = self.sample_num_ctx_trg(seed=seed)

        # Sample entire batch (context and target points)
        seed, batch = self.sample_full_batch(
            seed=seed,
            num_ctx=num_ctx,
            num_trg=num_trg,
        )

        return seed, Batch(
            x_ctx=batch.x[:, :num_ctx, :],
            y_ctx=batch.y[:, :num_ctx, :],
            x_trg=batch.x[:, num_ctx:, :],
            y_trg=batch.y[:, num_ctx:, :],
        )

    def sample_num_ctx_trg(self, seed: Seed) -> Tuple[Seed, tf.Tensor, tf.Tensor]:
        """Sample the numbers of context and target points in the dataset.

        Arguments:
            seed: Random seed.

        Returns:
            seed: Random seed generated by splitting.
            num_ctx: Number of context points.
            num_trg: Number of target points.
        """

        # Sample number of context points
        seed, num_ctx = randint(
            shape=(),
            seed=seed,
            minval=self.min_num_ctx,
            maxval=self.max_num_ctx,
        )

        # Sample number of target points
        seed, num_trg = randint(
            shape=(),
            seed=seed,
            minval=self.min_num_trg,
            maxval=self.max_num_trg,
        )

        return seed, num_ctx, num_trg

    def sample_full_batch(self, seed: Seed, num_ctx: int, num_trg: int) -> Tuple[Seed, Batch]:
        
        # Sample inputs, then outputs given inputs
        seed, x = self.sample_inputs(seed=seed, num_ctx=num_ctx, num_trg=num_trg)
        seed, y = self.sample_outputs(seed=seed, x=x)

        return seed, Batch(x=x, y=y)

    def sample_inputs(self, seed: Seed, num_ctx: int, num_trg: int) -> Tuple[Seed, tf.Tensor]:
        """Sample context and target inputs, sampled uniformly from the boxes
        defined by `context_range` and `target_range` respectively.

        Arguments:
            seed: Random seed.
            num_ctx: Number of context points.
            num_trg: Number of target points.

        Returns:
            seed: Random seed generated by splitting.
            x: Tensor of shape (batch_size, num_ctx + num_trg, dim) containing
                the context and target inputs.
        """
        
        seed, x_ctx = randu(
            shape=(self.batch_size, num_ctx, self.dim),
            seed=seed,
            minval=self.context_range[:, 0],
            maxval=self.context_range[:, 1],
        )

        seed, x_trg = randu(
            shape=(self.batch_size, num_trg, self.dim),
            seed=seed,
            minval=self.target_range[:, 0],
            maxval=self.target_range[:, 1],
        )

        return seed, tf.concat([x_ctx, x_trg], axis=1)

    @abstractmethod
    def sample_outputs(self, seed: Seed, x: tf.Tensor) -> Tuple[Seed, tf.Tensor]:
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
        pass