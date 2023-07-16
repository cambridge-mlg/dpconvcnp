
import tensorflow as tf
from tqdm import tqdm

from utils import initialize_experiment, train_epoch, valid_step
from dpconvcnp.utils import to_tensor, i32


def main():

    experiment, path, writer = initialize_experiment()

    dpconvcnp = experiment.model
    gen_train = experiment.generators.train
    gen_valid = experiment.generators.valid
    optimizer = experiment.optimizer
    seed = to_tensor(experiment.params.training_seed, i32)
    validation_seed = to_tensor(experiment.params.training_seed, i32)
    step = 0

    for epoch in range(100):
        
        seed, step = train_epoch(
            seed=seed,
            model=dpconvcnp,
            generator=gen_train,
            optimizer=optimizer,
            writer=writer,
            step=step,
        )


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

        _, result = valid_step(
            seed=validation_seed,
            model=dpconvcnp,
            generator=gen_valid,
        )
        writer.add_scalar("kl_diag", result["kl_diag"], epoch)


if __name__ == "__main__":
    main()
