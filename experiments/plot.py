from typing import Tuple, List, Union, Optional
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from argparse import Namespace

from dpconvcnp.random import Seed
from dpconvcnp.data.data import Batch
from dpconvcnp.utils import to_tensor, f32
from utils import get_batch_info
import dpsgp
import ray

tfd = tfp.distributions

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    path: str,
    model: Union[tf.Module, None],
    seed: Seed,
    batches: List[Batch],
    epoch: int = 0,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 512,
    params: Optional[Namespace] = None,
):
    # Get dimension of input data
    dim = batches[0].x_ctx.shape[-1]

    # If path for figures does not exist, create it
    os.makedirs(f"{path}/fig", exist_ok=True)

    if dim == 1:
        x_plot = np.linspace(x_range[0], x_range[1], points_per_dim)[None, :, None]
        x_plot = to_tensor(x_plot, f32)

        for i in range(num_fig):
            if model is None:
                xc = torch.as_tensor(
                    batches[i].x_ctx[0, ...].numpy(), dtype=torch.float64
                )
                yc = torch.as_tensor(
                    batches[i].y_ctx[0, ...].numpy(), dtype=torch.float64
                )
                xt = torch.as_tensor(
                    batches[i].x_trg[0, ...].numpy(), dtype=torch.float64
                )
                yt = torch.as_tensor(
                    batches[i].y_trg[0, ...].numpy(), dtype=torch.float64
                )

                epsilon = batches[i].epsilon[0].numpy()
                delta = batches[i].delta[0].numpy()

                params.epsilon = epsilon
                params.delta = delta

                elbo, model = dpsgp.utils.dp_train_model(xc, yc, params)

                model.eval()
                x_plot_pt = torch.as_tensor(x_plot[0, ...].numpy(), dtype=torch.float64)
                with torch.no_grad():
                    qf_params = model(xt)
                    qf_loc, qf_std = qf_params[:, : yt.shape[-1]], qf_params[
                        :, yt.shape[-1] :
                    ].pow(0.5)
                    qf = torch.distributions.Normal(qf_loc, qf_std)

                    model_nll = -model.likelihood.expected_log_prob(yt, qf).mean()

                    qf_params = model(x_plot_pt)
                    mean, std = qf_params[:, : yt.shape[-1]], qf_params[
                        :, yt.shape[-1] :
                    ].pow(0.5)

                    # Dimension padding for indexing later.
                    mean = mean.unsqueeze(0)
                    std = std.unsqueeze(0)
                    std = (std.pow(2) + model.likelihood.noise.pow(2)).pow(0.5)

                # Reset to None for logic.
                model = None

            else:
                # Use model to make predictions at x_plot
                _, mean, std = model(
                    seed=seed,
                    x_ctx=batches[i].x_ctx[:1],
                    y_ctx=batches[i].y_ctx[:1],
                    x_trg=x_plot[:1],
                    epsilon=batches[i].epsilon[:1],
                    delta=batches[i].delta[:1],
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
                model_nll = tf.reduce_sum(loss) / batches[i].y_trg.shape[1]

            # Use ground truth to make predictions at x_plot
            gt_mean, gt_std, _ = batches[i].gt_pred(
                x_ctx=batches[i].x_ctx[:1],
                y_ctx=batches[i].y_ctx[:1],
                x_trg=x_plot[:1],
            )

            _, _, gt_log_prob = batches[i].gt_pred(
                x_ctx=batches[i].x_ctx[:1],
                y_ctx=batches[i].y_ctx[:1],
                x_trg=batches[i].x_trg[:1],
                y_trg=batches[i].y_trg[:1],
            )
            gt_nll = tf.reduce_mean(-gt_log_prob) / batches[i].y_trg.shape[1]

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
                f"$\\delta$ = {info['delta']:.3f}\n"
                f"NLL = {model_nll:.3f} \t"
                f"GT NLL = {gt_nll:.3f}",
                fontsize=24,
            )

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.legend(loc="upper right", fontsize=14)
            plt.savefig(f"{path}/fig/epoch-{epoch:04d}-{i:03d}.png")
            plt.close()

    else:
        raise NotImplementedError
