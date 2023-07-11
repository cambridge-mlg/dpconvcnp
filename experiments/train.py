from typing import Tuple
import argparse
import os
import shutil

import tensorflow as tf
import tensorboard

from dpconvcnp.model.dpconvcnp import DPConvCNP
from dpconvcnp.data.gp import RandomScaleGPGenerator
from dpconvcnp.random import Seed
from dpconvcnp.data.data import Batch

log10 = tf.experimental.numpy.log10
f32 = tf.float32

# tf.debugging.enable_check_numerics()


def train_step(
        seed: Seed,
        model: tf.Module,
        batch: Batch,
        optimiser: tf.optimizers.Optimizer,
        norm_factor: float,
    ) -> Tuple[Seed, tf.Tensor]:

    with tf.GradientTape() as tape:
        seed, loss = model.loss(
            seed=seed,
            x_ctx=batch.x_ctx,
            y_ctx=batch.y_ctx,
            x_trg=batch.x_trg,
            y_trg=batch.y_trg,
            epsilon=batch.epsilon,
            delta=batch.delta,
        )
        loss = tf.reduce_mean(loss) / norm_factor

    gradients = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    return seed, loss


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
    default=None,
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
    default=1,
    help="Minimum number of context points.",
)

parser.add_argument(
    "--max-num-ctx",
    type=int,
    default=16,
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

parser.add_argument(
    "--points-per-unit",
    type=int,
    default=32,
    help="Number of points per unit.",
)

parser.add_argument(
    "--margin",
    type=float,
    default=0.1,
    help="Margin.",
)

parser.add_argument(
    "--lengthscale-init",
    type=float,
    default=0.1,
    help="Initial lengthscale.",
)

parser.add_argument(
    "--y-bound-init",
    type=float,
    default=2.,
    help="Initial y bound.",
)

parser.add_argument(
    "--w-noise-init",
    type=float,
    default=0.5,
    help="Initial w noise.",
)

parser.add_argument(
    "--encoder-lengthscale-trainable",
    action="store_true",
    help="Whether to train the encoder lengthscale.",
)

parser.add_argument(
    "--y-bound-trainable",
    action="store_true",
    help="Whether to train the y bound.",
)

parser.add_argument(
    "--w-noise-trainable",
    action="store_true",
    help="Whether to train the w noise.",
)

parser.add_argument(
    "--architecture",
    type=str,
    choices=[
        "unet",
    ],
    default="unet",
    help="Convolutional architecture to use.",
)

parser.add_argument(
    "--first-channels",
    type=int,
    default=32,
    help="Number of channels in the first convolutional layer.",
)

parser.add_argument(
    "--last-channels",
    type=int,
    default=2,
    help="Number of channels in the last convolutional layer.",
)

parser.add_argument(
    "--kernel-size",
    type=int,
    default=5,
    help="Size of the convolutional kernels.",
)

parser.add_argument(
    "--num-channels",
    type=int,
    nargs="+",
    default=[64, 64, 64, 64, 64, 64],
    help="Number of channels in each UNet layer.",
)

parser.add_argument(
    "--strides",
    type=int,
    nargs="+",
    default=[2, 2, 2, 2, 2, 2],
    help="Strides in each UNet layer.",
)

parser.add_argument(
    "--dim",
    type=int,
    default=1,
    help="Dimensionality of the input data.",
)

parser.add_argument(
    "--model-seed",
    type=int,
    default=0,
    help="Random seed.",
)

parser.add_argument(
    "--experiment-seed",
    type=int,
    default=1,
    help="Random seed.",
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-4,
    help="Learning rate.",
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

architecture_kwargs = {
    "first_channels": args.first_channels,
    "last_channels": args.last_channels,
    "kernel_size": args.kernel_size,
    "num_channels": args.num_channels,
    "strides": args.strides,
    "dim": args.x_dim,
    "seed": args.model_seed,
}

dpconvcnp = DPConvCNP(
    points_per_unit=args.points_per_unit,
    margin=args.margin,
    lenghtscale_init=args.lengthscale_init,
    y_bound_init=args.y_bound_init,
    w_noise_init=args.w_noise_init,
    encoder_lengthscale_trainable=args.encoder_lengthscale_trainable,
    y_bound_trainable=args.y_bound_trainable,
    w_noise_trainable=args.w_noise_trainable,
    architcture=args.architecture,
    architcture_kwargs=architecture_kwargs,
)

tmp_logdir = "logs"
#if os.path.exists("logs"):
#    shutil.rmtree(tmp_logdir)

writer = tensorboard.summary.Writer(tmp_logdir)

seed = [0, args.experiment_seed]

optimiser = tf.optimizers.Adam(learning_rate=args.learning_rate)

for epoch in range(100):
    for i, batch in enumerate(generator):
        seed, loss = train_step(
            seed=seed,
            model=dpconvcnp,
            batch=batch,
            optimiser=optimiser,
            norm_factor=args.max_num_trg,
        )

        writer.add_scalar("loss", loss, i)

        if i % 100 == 0:

            x_ctx = batch.x_ctx[:1]
            y_ctx = batch.y_ctx[:1]
            x_trg = tf.linspace(-4., 4., 400)[None, :, None]
            epsilon = batch.epsilon[:1]
            delta = batch.delta[:1]

            seed, mean, std = dpconvcnp(
                seed=seed,
                x_ctx=x_ctx,
                y_ctx=y_ctx,
                x_trg=x_trg,
                epsilon=epsilon,
                delta=delta,
            )
            #print(i, dpconvcnp.dpsetconv_encoder.lengthscale)
            print(epoch, i, loss)
            #print(i, -tf.reduce_mean(batch.gt_log_lik / args.max_num_trg))

            import matplotlib.pyplot as plt
            import numpy as np
            
            idx = np.argsort(x_trg[0, :, 0])
            plt.scatter(batch.x_ctx.numpy()[0, :, 0], batch.y_ctx.numpy()[0, :, 0], c="k")
            plt.scatter(batch.x_trg.numpy()[0, :, 0], batch.y_trg.numpy()[0, :, 0], c="r")
            plt.plot(x_trg.numpy()[0, idx, 0], mean.numpy()[0, idx, 0], c="b")
            plt.fill_between(
                x_trg.numpy()[0, idx, 0],
                mean.numpy()[0, idx, 0] - 2. * std.numpy()[0, idx, 0],
                mean.numpy()[0, idx, 0] + 2. * std.numpy()[0, idx, 0],
                color="tab:blue",
                alpha=0.2,
            )

            plt.savefig(f"figs/{i}.png")
            plt.clf()
