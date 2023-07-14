
import tensorflow as tf
import tensorboard

from utils import initialize_experiment, train_step

log10 = tf.experimental.numpy.log10
f32 = tf.float32


def main():

    experiment, path, writer = initialize_experiment()

    dpconvcnp = experiment.model
    generator = experiment.generator
    optimizer = experiment.optimizer
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

                #import matplotlib.pyplot as plt
                #import numpy as np

                #gt_mean, gt_std, _ = batch.gt_pred(
                #    x_ctx=x_ctx,
                #    y_ctx=y_ctx,
                #    x_trg=x_trg,
                #)
                
                #idx = np.argsort(x_trg[0, :, 0])
                #plt.scatter(batch.x_ctx.numpy()[0, :, 0], batch.y_ctx.numpy()[0, :, 0], c="k")
                #plt.scatter(batch.x_trg.numpy()[0, :, 0], batch.y_trg.numpy()[0, :, 0], c="r")
                #plt.plot(x_trg.numpy()[0, idx, 0], mean.numpy()[0, idx, 0], c="b")
                #plt.fill_between(
                #    x_trg.numpy()[0, idx, 0],
                #    mean.numpy()[0, idx, 0] - 2. * std.numpy()[0, idx, 0],
                #    mean.numpy()[0, idx, 0] + 2. * std.numpy()[0, idx, 0],
                #    color="tab:blue",
                #    alpha=0.2,
                #)
                #plt.plot(x_trg.numpy()[0, :, 0], gt_mean.numpy()[0, :], "--", color="tab:purple")
                #plt.plot(x_trg.numpy()[0, :, 0], gt_mean.numpy()[0, :] + 2 * gt_std.numpy()[0, :], "--", color="tab:purple")
                #plt.plot(x_trg.numpy()[0, :, 0], gt_mean.numpy()[0, :] - 2 * gt_std.numpy()[0, :], "--", color="tab:purple")

                #plt.savefig(f"figs/{i}.png")
                #plt.clf()


if __name__ == "__main__":
    main()
