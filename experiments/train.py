from typing import Tuple
import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf
import tensorflow as tf
import tensorboard

from dpconvcnp.random import Seed
from dpconvcnp.data.data import Batch
from dpconvcnp.utils import make_seed

log10 = tf.experimental.numpy.log10
f32 = tf.float32


def train_step(
        seed: Seed,
        model: tf.Module,
        batch: Batch,
        optimizer: tf.optimizers.Optimizer,
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
        loss = tf.reduce_mean(loss) / batch.y_trg.shape[1]

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return seed, loss


parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str)

args = parser.parse_args()

experiment = instantiate(OmegaConf.load(args.config))

dpconvcnp = experiment.model
generator = experiment.generator
optimizer = experiment.optimizer

tmp_logdir = "scratch"

writer = tensorboard.summary.Writer(tmp_logdir)

seed = experiment.params.experiment_seed


c = 0
for epoch in range(100):
    for i, batch in enumerate(generator):
        c += 1
        seed, loss = train_step(
            seed=seed,
            model=dpconvcnp,
            batch=batch,
            optimizer=optimizer,
        )

        writer.add_scalar("loss", loss, c)
        writer.add_scalar("lengthscale", dpconvcnp.dpsetconv_encoder.lengthscale, c)
        writer.add_scalar("y_bound", dpconvcnp.dpsetconv_encoder.y_bound, c)
        writer.add_scalar("w_noise", dpconvcnp.dpsetconv_encoder.w_noise, c)

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

            gt_mean, gt_std, _ = batch.gt_pred(
                x_ctx=x_ctx,
                y_ctx=y_ctx,
                x_trg=x_trg,
            )
            
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
            plt.plot(x_trg.numpy()[0, :, 0], gt_mean.numpy()[0, :], "--", color="tab:purple")
            plt.plot(x_trg.numpy()[0, :, 0], gt_mean.numpy()[0, :] + 2 * gt_std.numpy()[0, :], "--", color="tab:purple")
            plt.plot(x_trg.numpy()[0, :, 0], gt_mean.numpy()[0, :] - 2 * gt_std.numpy()[0, :], "--", color="tab:purple")


            plt.savefig(f"figs/{i}.png")
            plt.clf()
