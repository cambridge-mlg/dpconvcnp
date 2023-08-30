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

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def plot(
    path: str,
    model: tf.Module,
    seed: Seed,
    batches: List[Batch],
    epoch: int = 0,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8., 6.),
    x_range: Tuple[float, float] = (-5., 5.),
    y_lim: Tuple[float, float] = (-3., 3.),
    points_per_dim: int = 512,
):

    # Get dimension of input data
    dim = batches[0].x_ctx.shape[-1]

    # If path for figures does not exist, create it
    os.makedirs(f"{path}/fig", exist_ok=True)

    if dim == 1:

        x_plot = np.linspace(x_range[0], x_range[1], points_per_dim)[None, :, None]
        x_plot = to_tensor(x_plot, f32)

        for i in range(num_fig):

            # Use model to make predictions at x_plot
            seed, mean, std = model(
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
                mean[0, :, 0] - 2. * std[0, :, 0],
                mean[0, :, 0] + 2. * std[0, :, 0],
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
        raise NotImplementedError
    
