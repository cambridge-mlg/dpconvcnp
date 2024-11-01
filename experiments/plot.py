from typing import Optional, Dict, Tuple, List
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
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    points_per_dim: int = 128,
    plot_options: Optional[Dict] = None,
):
    # Get dimension of input data
    dim = batches[0].x_ctx.shape[-1]

    # If path for figures does not exist, create it
    os.makedirs(f"{path}/fig", exist_ok=True)

    if dim == 1:
        x_plot = np.linspace(
            plot_options.xlim[0],
            plot_options.xlim[1],
            points_per_dim,
        )[None, :, None]
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

            if batches[i].gt_pred is not None:
                # Use ground truth to make predictions at x_plot
                gt_mean, gt_std, _, _ = batches[i].gt_pred(
                    x_ctx=batches[i].x_ctx[:1],
                    y_ctx=batches[i].y_ctx[:1],
                    x_trg=x_plot[:1],
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
            plt.xlim(plot_options.xlim)
            plt.ylim(plot_options.ylim)

            # Set title
            info = get_batch_info(batches[i], 0)
            ell_str = (
                f"$\\ell$ = {info['lengthscale']:.2f}  "
                if info["lengthscale"]
                else ""
            )
            plt.title(
                f"$N = {info['n']}$   "
                f"{ell_str}"
                f"$\\epsilon$ = {info['epsilon']:.2f}  "
                f"$\\delta$ = {info['delta']:.5f}",
                fontsize=24,
            )

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.legend(loc="upper right", fontsize=14)
            plt.savefig(f"{path}/fig/epoch-{epoch:04d}-{i:03d}.png")
            plt.close()

    else:
        x_plot = np.linspace(
            plot_options.xlim[0] - 1e-1,
            plot_options.xlim[1] + 1e-1,
            points_per_dim,
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
                vmin=-1.0,
                vmax=1.0,
                edgecolors="k",
                linewidths=0.5,
            )

            plt.contourf(
                x_plot[0, ..., 0],
                x_plot[0, ..., 1],
                mean,
                cmap="coolwarm",
                zorder=1,
                vmin=-1.0,
                vmax=1.0,
            )

            plt.xlim(plot_options.xlim)
            plt.ylim(plot_options.xlim)

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
                gt_mean, gt_std, _, _ = gt_pred(
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

            centering = gt_mean_ordered if gt_pred is not None else 0.0

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
