from abc import ABC, abstractmethod
from typing import List, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class DataGenerator(ABC):

    def __init__(
            self,
            *,
            seed: tf.Tensor,
            samples_per_epoch: int,
            batch_size: int,
        ):
        """Base data generator, which can be used to derive other data generators,
        such as synthetic generators or real data generators.

        Arguments:
            seed: Random seed for generator.
            samples_per_epoch: Number of samples per epoch.
            batch_size: Batch size.
        """

        # Set random seed for generator
        self.seed = seed

        # Set generator parameters
        self.samples_per_epoch = samples_per_epoch
        self.num_batches = samples_per_epoch // batch_size + 1
        self.batch_size = batch_size

        # Set epoch counter
        self.i = 0

    def __iter__(self):
        """Reset epoch counter and return self."""
        self.i = 0
        return self

    def __next__(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """
        
        if self.i >= self.num_batches:
            raise StopIteration
        
        else:
            self.i += 1
            self.seed, batch = self.generate_batch(seed=self.seed)
            return batch

    def generate_batch(
            self,
            seed: tf.Tensor,
        ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]:
        """Generate batch of data using the random seed `seed`.

        Arguments:
            seed: Random seed for batch.
        
        Returns:
            seed: Random seed for next batch.
            batch: Tuple of tensors containing the context and target data.
        """

        # Create full batch with context and target data not yet split
        seed, batch_ctx_trg = self.generate_full_batch(seed=seed)

        return self.split_to_context_and_target(seed=seed, batch_ctx_trg=batch_ctx_trg)

    @abstractmethod
    def generate_full_batch(
            self,
            seed: tf.Tensor,
        ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def split_to_context_and_target(
            self,
            seed: tf.Tensor,
            batch_ctx_trg: Tuple[tf.Tensor, tf.Tensor],
        ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        raise NotImplementedError


class SyntheticGenerator(DataGenerator):
    
    def __init__(
        *,
        self,
        min_num_trg: tfd.distribution,
        max_num_trg: tfd.distribution,
        context_range: List[Tuple[int, int]],
        target_range: List[Tuple[int, int]],
        **kwargs,
    ):

        super().__init__(**kwargs)
        
        # Set synthetic generator parameters
        self.min_num_trg = min_num_trg
        self.max_num_trg = max_num_trg
        self.context_range = context_range
        self.target_range = target_range

class GPGenerator(SyntheticGenerator):
    pass
