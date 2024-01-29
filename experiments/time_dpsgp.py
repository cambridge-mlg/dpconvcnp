from utils import initialize_experiment
from functools import partial

import time
import torch
import dpsgp
import wandb
import numpy as np
import argparse
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from dpconvcnp.data.data import Batch

torch.set_default_dtype(torch.float64)


def train_model(batch: Batch, params: argparse.Namespace):
    # Convert everything to tensors.
    xc = torch.as_tensor(batch.x_ctx[0, ...].numpy(), dtype=torch.float64)
    yc = torch.as_tensor(batch.y_ctx[0, ...].numpy(), dtype=torch.float64)
    xt = torch.as_tensor(batch.x_trg[0, ...].numpy(), dtype=torch.float64)
    yt = torch.as_tensor(batch.y_trg[0, ...].numpy(), dtype=torch.float64)

    # Get batch epislon and delta.
    params.epsilon = batch.epsilon[0].numpy()
    params.delta = batch.delta[0].numpy()

    # Train model and get results.
    elbo, model, tdict = dpsgp.utils.dp_train_model(xc, yc, params)
    elbo = elbo.detach()

    # Get predictions!
    with torch.no_grad():
        qf_params = model(xt)
        mean, std = qf_params[:, : yt.shape[-1]], qf_params[:, yt.shape[-1] :].pow(0.5)
        pred_std = (std.pow(2) + model.likelihood.noise.pow(2)).pow(0.5)

        qf = torch.distributions.Normal(mean, std)
        qf_pred = torch.distributions.Normal(mean, pred_std)

        nll = -qf_pred.log_prob(yt).mean()
        exp_ll = -model.likelihood.expected_log_prob(yt, qf).mean()
        rmse = (mean - yt).pow(2).mean().sqrt()
        kl = model._module.kl_divergence()

    result = {
        "batch": batch,
        "elbo": elbo,
        "exp_ll": exp_ll,
        "kl": kl,
        "nll": nll,
        "mean": mean,
        "std": std,
        "rmse": rmse,
        "tdelta_train_and_inference": tdict["train_and_inference"],
        "tdelta_train": tdict["train"],
        "tdelta_inference": tdict["inference"],
    }

    if batch.gt_pred is not None:
        gt_mean, gt_std, gt_log_lik, _ = batch.gt_pred(
            x_ctx=batch.x_ctx,
            y_ctx=batch.y_ctx,
            x_trg=batch.x_trg,
            y_trg=batch.y_trg,
        )

        gt_loss = -gt_log_lik / batch.y_trg.shape[1]

        result["gt_mean"] = gt_mean
        result["gt_loss"] = gt_loss
        result["gt_std"] = gt_std

    return result


def main():
    experiment, config, _, _, _, _ = initialize_experiment()

    gen_train = experiment.generators.train

    wandb.init(
        project=experiment.misc.project,
        name=experiment.misc.name,
        config=config,
    )

    total_times = train_times = train_and_inference_times = inference_times = []
    for batch in gen_train:
        params = experiment.params
        params.epsilon = batch.epsilon[0]
        params.delta = batch.delta[0]

        t0 = time.time()
        result = train_model(batch, params)
        t1 = time.time()

        tdelta = t1 - t0
        wandb.log({"total_time": tdelta})
        wandb.log({"train_and_inference_time": result["tdelta_train_and_inference"]})
        wandb.log({"train_time": result["tdelta_train"]})
        wandb.log({"inference_time": result["tdelta_inference"]})

        train_and_inference_times.append(result["tdelta_train_and_inference"])
        train_times.append(result["tdelta_train"])
        inference_times.append(result["tdelta_inference"])
        total_times.append(tdelta)

    for name, time_list in zip(
        ("total", "train_and_inference", "train", "inference"),
        (total_times, train_and_inference_times, train_times, inference_times),
    ):
        times = np.array(time_list)
        wandb.run.summary[f"mean_{name}_time"] = times.mean()
        wandb.run.summary[f"std_{name}_time"] = times.std()
        wandb.run.summary[f"ste_{name}_time"] = times.std() / (len(times) ** 0.5)


if __name__ == "__main__":
    main()
