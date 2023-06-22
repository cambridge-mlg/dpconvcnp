from typing import List, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class DataGenerator:

    def __init__(
            self,
            *,
            seed: tf.Tensor,
            samples_per_epoch: int,
            batch_size: int,
        ):

        # Set random seed for generator
        self.seed = seed

        # Set generator parameters
        self.samples_per_epoch = samples_per_epoch
        self.num_batches = samples_per_epoch // batch_size + 1
        self.batch_size = batch_size

        # Set epoch counter
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        
        if self.i >= self.num_batches:
            self.i = 0
            raise StopIteration
        
        else:
            self.i += 1
            self.state, batch = self.generate_batch(state=self.state)
            return batch

    def generate_batch(
            self,
            state: tf.Tensor,
        ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

        self.self, (num_ctx, num_trg) = self.draw_nums_context_and_target(
            state=state,
        )

        self.state, context_target_batch = self.generate_context_target_batch(
            state=state,
            num_ctx=num_ctx,
            num_trg=num_trg,
        )

        raise NotImplementedError

    def generate_context_target_batch(
            self,
            state: tf.Tensor,
            num_ctx: int,
            num_trg: int,
        ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        raise NotImplementedError

    def draw_nums_context_and_target(
            self,
            state: tf.Tensor,
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

        self.min_num_trg = min_num_trg
        self.max_num_trg = max_num_trg
        self.context_range = context_range
        self.target_range = target_range

class GPGenerator(SyntheticGenerator):
    pass
