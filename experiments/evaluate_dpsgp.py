import os
from typing import List, Dict

from utils import get_batch_info
from plot_dpsgp import plot

import argparse
import optuna
import dpsgp
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import ray
import math
import wandb

from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm.auto import tqdm
from dpconvcnp.data.data import Batch


@ray.remote
def dp_train_model(batch: Batch, params: argparse.Namespace):
    # Convert everything to tensors.
    xc = torch.as_tensor(batch.x_ctx[0, ...].numpy(), dtype=torch.float64)
    yc = torch.as_tensor(batch.y_ctx[0, ...].numpy(), dtype=torch.float64)
    xt = torch.as_tensor(batch.x_trg[0, ...].numpy(), dtype=torch.float64)
    yt = torch.as_tensor(batch.y_trg[0, ...].numpy(), dtype=torch.float64)

    # Get batch epislon and delta.
    params.epsilon = batch.epsilon[0].numpy()
    params.delta = batch.delta[0].numpy()

    # Train model and get results.
    elbo, model = dpsgp.utils.dp_train_model(xc, yc, params)
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

    gt_mean, gt_std, gt_log_lik = batch.gt_pred(
        x_ctx=batch.x_ctx,
        y_ctx=batch.y_ctx,
        x_trg=batch.x_trg,
        y_trg=batch.y_trg,
    )

    gt_loss = -gt_log_lik / batch.y_trg.shape[1]

    return {
        "batch": batch,
        "elbo": elbo,
        "exp_ll": exp_ll,
        "kl": kl,
        "nll": nll,
        "mean": mean,
        "std": std,
        "gt_mean": gt_mean,
        "gt_std": gt_std,
        "gt_loss": gt_loss,
        "rmse": rmse,
    }


def validate_dpsgp(generator, params: argparse.Namespace):
    batch_results = []
    for batch in tqdm(generator, total=generator.num_batches, desc="Validation"):
        batch_results.append(dp_train_model.remote(batch, params))

    batch_results = [ray.get(batch_result) for batch_result in batch_results]

    result = {
        k: [batch_result[k] for batch_result in batch_results]
        for k in batch_results[0].keys()
    }
    for k in ["elbo", "nll", "exp_ll", "kl", "rmse"]:
        result[k] = torch.stack(result[k])

    result["gt_loss"] = tf.concat(result["gt_loss"], axis=0)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--evaluation_config", type=str)
    parser.add_argument("--lengthscale", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--num_ctx", type=int)
    args, config_changes = parser.parse_known_args()

    api = wandb.Api()
    run = api.run(args.run_path)
    config = run.config

    # Initialise wandb.
    wandb.init(
        resume="must",
        id=run.id,
        project=run.project,
        name=run.name,
    )

    # Update config with evaluation_config and config_changes.
    evaluation_config = OmegaConf.load(args.evaluation_config)
    config_changes = OmegaConf.from_cli(config_changes)
    config = OmegaConf.merge(config, evaluation_config)
    config = OmegaConf.merge(config, config_changes)

    # Assign before instantiation.
    evaluation_config.generator.min_log10_lengthscale = math.log10(args.lengthscale)
    evaluation_config.generator.max_log10_lengthscale = math.log10(args.lengthscale)
    evaluation_config.generator.min_epsilon = args.epsilon
    evaluation_config.generator.max_epsilon = args.epsilon
    evaluation_config.generator.min_num_ctx = args.num_ctx
    evaluation_config.generator.max_num_ctx = args.num_ctx
    evaluation_config.params.eval_name = f"lengthscale-{args.lengthscale}/eps-{args.epsilon}/log10delta-{evaluation_config.generator.min_log10_delta}/nc-{args.num_ctx}"
    evaluation = instantiate(evaluation_config)

    # Set lengthscale, epsilon and num_ctx of the generator.

    generator = evaluation.generator
    batches = [batch for batch in generator]

    # Load best parameters from wandb run.
    best_params = run.summary.best_params
    params = argparse.Namespace(**best_params)

    plot(
        model=None,
        seed=list(evaluation.params.evaluation_seed),
        batches=batches,
        params=params,
        num_fig=5,
        name=f"eval/{evaluation.params.eval_name}",
    )

    result = validate_dpsgp(generator, params)
    result = {
        k: result[k].numpy() for k in ["elbo", "gt_loss", "nll", "exp_ll", "kl", "rmse"]
    }

    # Log summary of evaluation.
    for k in result.keys():
        wandb.run.summary[f"eval/{evaluation.params.eval_name}/mean_{k}"] = result[
            k
        ].mean()
        wandb.run.summary[f"eval/{evaluation.params.eval_name}/std_{k}"] = result[
            k
        ].std()


if __name__ == "__main__":
    main()
