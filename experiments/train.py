import argparse

import tensorflow as tf

from dpconvcnp.data.gp import (
    KERNEL_TYPES,
    RandomScaleGPGenerator,
)

log10 = tf.experimental.numpy.log10
f32 = tf.float32

parser = argparse.ArgumentParser(
    description="Train a DPConvCNP model on a dataset.",
)

parser.add_argument(
    "--data-seed",
    type=int,
    default=0,
    help="Random seed for the data generator.",
)

parser.add_argument(
    "--data",
    type=str,
    choices=[
        "eq",
        "matern12",
        "matern32",
        "matern52",
        "noisy_mixture",
        "weakly_periodic",
    ],
)

parser.add_argument(
    "--x-dim",
    type=int,
    default=1,
    help="Dimensionality of the inputs.",
)

parser.add_argument(
    "--min-log10-lengthscale",
    type=float,
    default=log10(0.25),
    help="Minimum log10 lengthscale.",
)

parser.add_argument(
    "--max-log10-lengthscale",
    type=float,
    default=log10(0.25),
    help="Maximum log10 lengthscale.",
)

parser.add_argument(
    "--noise-std",
    type=float,
    default=0.1,
    help="Standard deviation of the noise.",
)

parser.add_argument(
    "--min-num-ctx",
    type=int,
    default=0,
    help="Minimum number of context points.",
)

parser.add_argument(
    "--max-num-ctx",
    type=int,
    default=128,
    help="Maximum number of context points.",
)

parser.add_argument(
    "--min-num-trg",
    type=int,
    default=128,
    help="Minimum number of target points.",
)

parser.add_argument(
    "--max-num-trg",
    type=int,
    default=128,
    help="Maximum number of target points.",
)

parser.add_argument(
    "--context-range",
    type=float,
    nargs=2,
    default=[-2., 2.],
    help="Range of context points.",
)

parser.add_argument(
    "--target-range",
    type=float,
    nargs=2,
    default=[-4., 4.],
    help="Range of target points.",
)

parser.add_argument(
    "--samples-per-epoch",
    type=int,
    default=2**16,
    help="Number of samples per epoch.",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size.",
)

parser.add_argument(
    "--min-epsilon",
    type=float,
    default=3.,
    help="Minimum DP epsilon.",
)

parser.add_argument(
    "--max-epsilon",
    type=float,
    default=3.,
    help="Maximum DP epsilon.",
)

parser.add_argument(
    "--min-log10-delta",
    type=float,
    default=log10(1e-3),
    help="Minimum log10 DP delta.",
)

parser.add_argument(
    "--max-log10-delta",
    type=float,
    default=log10(1e-3),
    help="Maximum log10 DP delta.",
)

args = parser.parse_args()

data_seed = [0, args.data_seed]

generator = RandomScaleGPGenerator(
    seed=data_seed,
    dim=args.x_dim,
    kernel_type=args.data,
    min_log10_lengthscale=args.min_log10_lengthscale,
    max_log10_lengthscale=args.max_log10_lengthscale,
    noise_std=args.noise_std,
    min_num_ctx=args.min_num_ctx,
    max_num_ctx=args.max_num_ctx,
    min_num_trg=args.min_num_trg,
    max_num_trg=args.max_num_trg,
    context_range=tf.convert_to_tensor(args.x_dim*[args.context_range], dtype=f32),
    target_range=tf.convert_to_tensor(args.x_dim*[args.target_range], dtype=f32),
    samples_per_epoch=args.samples_per_epoch,
    batch_size=args.batch_size,
    min_epsilon=args.min_epsilon,
    max_epsilon=args.max_epsilon,
    min_log10_delta=args.min_log10_delta,
    max_log10_delta=args.max_log10_delta,
)

for batch in generator:
    print(batch)
    breakpoint()
