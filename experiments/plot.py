from typing import Tuple, List
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from dpconvcnp.random import Seed
from dpconvcnp.data.data import Batch
from dpconvcnp.utils import to_tensor, f32
from utils import get_batch_info

tfd = tfp.distributions

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    path: str,
    model: tf.Module,
    seed: Seed,
    batches: List[Batch],
    epoch: int = 0,
    num_fig: int = 3,
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Tuple[float, float] = (-1.1, 1.1),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 128,
):
    # Get dimension of input data
    dim = batches[0].x_ctx.shape[-1]

    # If path for figures does not exist, create it
    os.makedirs(f"{path}/fig", exist_ok=True)

    if dim == 1:
        x_plot = np.linspace(x_range[0], x_range[1], points_per_dim)[
            None, :, None
        ]
        x_plot = to_tensor(x_plot, f32)

        for i in range(num_fig):
            # Use model to make predictions at x_plot
            _, mean, std = model(
                seed=seed,
                x_ctx=batches[i].x_ctx[:1],
                y_ctx=batches[i].y_ctx[:1],
                x_trg=x_plot[:1],
                epsilon=batches[i].epsilon[:1],
                delta=batches[i].delta[:1],
            )

            # Use ground truth to make predictions at x_plot
            gt_mean, gt_std, _ = batches[i].gt_pred(
                x_ctx=batches[i].x_ctx[:1],
                y_ctx=batches[i].y_ctx[:1],
                x_trg=x_plot[:1],
            )

            seed, loss, _, _ = model.loss(
                seed=seed,
                x_ctx=batches[i].x_ctx[:1],
                y_ctx=batches[i].y_ctx[:1],
                x_trg=batches[i].x_trg[:1],
                y_trg=batches[i].y_trg[:1],
                epsilon=batches[i].epsilon[:1],
                delta=batches[i].delta[:1],
            )

            # Make figure for plotting
            plt.figure(figsize=figsize)

            # Plot context and target points
            plt.scatter(
                batches[i].x_ctx[0, :, 0],
                batches[i].y_ctx[0, :, 0],
                c="k",
                label="Context",
                s=20,
            )

            plt.scatter(
                batches[i].x_trg[0, :, 0],
                batches[i].y_trg[0, :, 0],
                c="r",
                label="Target",
                s=20,
            )

            # Plot model predictions
            plt.plot(
                x_plot[0, :, 0],
                mean[0, :, 0],
                c="tab:blue",
            )

            plt.fill_between(
                x_plot[0, :, 0],
                mean[0, :, 0] - 2.0 * std[0, :, 0],
                mean[0, :, 0] + 2.0 * std[0, :, 0],
                color="tab:blue",
                alpha=0.2,
                label="Model",
            )

            # Plot ground truth
            plt.plot(
                x_plot[0, :, 0],
                gt_mean[0, :],
                "--",
                color="tab:purple",
            )

            plt.plot(
                x_plot[0, :, 0],
                gt_mean[0, :] + 2 * gt_std[0, :],
                "--",
                color="tab:purple",
            )

            plt.plot(
                x_plot[0, :, 0],
                gt_mean[0, :] - 2 * gt_std[0, :],
                "--",
                color="tab:purple",
                label="Ground truth",
            )

            # Set axis limits
            plt.xlim(x_range)
            plt.ylim(y_lim)

            # Set title
            info = get_batch_info(batches[i], 0)
            plt.title(
                f"$N = {info['n']}$   "
                f"$\\ell$ = {info['lengthscale']:.2f}  "
                f"$\\epsilon$ = {info['epsilon']:.2f}  "
                f"$N\\ell \\epsilon$ = {info['nle']:.0f}  "
                f"$\\delta$ = {info['delta']:.3f}",
                fontsize=24,
            )

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.legend(loc="upper right", fontsize=14)
            plt.savefig(f"{path}/fig/epoch-{epoch:04d}-{i:03d}.png")
            plt.close()

    else:
        x_plot = np.linspace(
            x_range[0] - 1e-1, x_range[1] + 1e-1, points_per_dim
        )
        x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)[None, ...]

        for i in range(num_fig):
            _, mean, std = model(
                seed=seed,
                x_ctx=batches[i].x_ctx[:1],
                y_ctx=batches[i].y_ctx[:1],
                x_trg=to_tensor(np.reshape(x_plot, (1, -1, 2)), f32),
                epsilon=batches[i].epsilon[:1],
                delta=batches[i].delta[:1],
            )

            mean = tf.reshape(mean, (1, points_per_dim, points_per_dim, 1))
            mean = mean[0, ..., 0]
            std = tf.reshape(std, (1, points_per_dim, points_per_dim, 1))
            std = std[0, ..., 0]

            plt.figure(figsize=(2 * figsize[0], figsize[1]))
            plt.subplot(1, 2, 1)

            plt.scatter(
                batches[i].x_ctx[0, :, 0],
                batches[i].x_ctx[0, :, 1],
                c=batches[i].y_ctx[0, :, 0],
                marker="o",
                s=20,
                cmap="coolwarm",
                zorder=2,
                vmin=-1.5,
                vmax=1.5,
                edgecolors="k",
                linewidths=0.5,
            )

            plt.contourf(
                x_plot[0, ..., 0],
                x_plot[0, ..., 1],
                mean,
                cmap="coolwarm",
                zorder=1,
                vmin=-1.5,
                vmax=1.5,
            )

            plt.xlim(x_range)
            plt.ylim(x_range)

            _, mean, std = model(
                seed=seed,
                x_ctx=batches[i].x_ctx[:1],
                y_ctx=batches[i].y_ctx[:1],
                x_trg=batches[i].x_trg[:1],
                epsilon=batches[i].epsilon[:1],
                delta=batches[i].delta[:1],
            )

            gt_pred = batches[i].gt_pred

            if gt_pred is not None:
                gt_mean, gt_std, _ = gt_pred(
                    x_ctx=batches[i].x_ctx[:1],
                    y_ctx=batches[i].y_ctx[:1],
                    x_trg=batches[i].x_trg[:1],
                )

            idx = (
                np.argsort(gt_std[0, :, 0].numpy())
                if gt_pred is not None
                else np.argsort(std[0, :, 0].numpy())
            )

            y_trg_ordered = batches[i].y_trg[0, :, 0].numpy()[idx]
            mean_ordered = mean[0, :, 0].numpy()[idx]
            std_ordered = std[0, :, 0].numpy()[idx]

            if gt_pred is not None:
                gt_mean_ordered = gt_mean[0, :, 0].numpy()[idx]
                gt_std_ordered = gt_std[0, :, 0].numpy()[idx]

            centering = (
                gt_mean_ordered if gt_pred is not None else 0.
            )

            plt.subplot(1, 2, 2)
            plt.scatter(
                np.arange(len(y_trg_ordered)),
                y_trg_ordered - centering,
                c="tab:red",
                marker="o",
                s=10,
                zorder=3,
            )

            plt.errorbar(
                np.arange(len(y_trg_ordered)),
                mean_ordered - centering,
                yerr=1.96 * std_ordered,
                c="black",
                fmt="",
                linestyle="",
                capsize=2,
                zorder=1,
            )

            if gt_pred is not None:
                plt.errorbar(
                    np.arange(len(y_trg_ordered)),
                    gt_mean_ordered - centering,
                    yerr=1.96 * gt_std_ordered,
                    c="tab:purple",
                    fmt="",
                    linestyle="",
                    capsize=2,
                    zorder=2,
                )
                plt.xticks([])
                plt.yticks([])

            plt.ylim([-1.5, 1.5])

            plt.savefig(f"{path}/fig/epoch-{epoch:04d}-{i:03d}.png")
            plt.close()
