import os
from typing import List, Dict

from utils import get_batch_info
from plot import plot

import argparse
import optuna
import dpsgp
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import ray
import math

from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm.auto import tqdm
from dpconvcnp.data.data import Batch


@ray.remote
def dp_train_model(batch, params):
    xc = torch.as_tensor(batch.x_ctx[0, ...].numpy(), dtype=torch.float64)
    yc = torch.as_tensor(batch.y_ctx[0, ...].numpy(), dtype=torch.float64)
    xt = torch.as_tensor(batch.x_trg[0, ...].numpy(), dtype=torch.float64)
    yt = torch.as_tensor(batch.y_trg[0, ...].numpy(), dtype=torch.float64)

    xt_ctx = xt[torch.nonzero(xt[:, 0].abs() <= 2.0, as_tuple=True)[0]]
    yt_ctx = yt[torch.nonzero(xt[:, 0].abs() <= 2.0, as_tuple=True)[0]]
    xt_trg = xt[torch.nonzero(xt[:, 0].abs() > 2.0, as_tuple=True)[0]]
    yt_trg = yt[torch.nonzero(xt[:, 0].abs() > 2.0, as_tuple=True)[0]]

    params.epsilon = batch.epsilon[0].numpy()
    params.delta = batch.delta[0].numpy()

    elbo, model = dpsgp.utils.dp_train_model(xc, yc, params)
    elbo = elbo.detach()

    with torch.no_grad():
        qf_params = model(xt)
        mean, std = qf_params[:, : yt.shape[-1]], qf_params[:, yt.shape[-1] :].pow(0.5)
        pred_std = (std.pow(2) + model.likelihood.noise.pow(2)).pow(0.5)
        qf = torch.distributions.Normal(mean, std)
        qf_pred = torch.distributions.Normal(mean, pred_std)
        nll = -qf_pred.log_prob(yt).mean()
        exp_ll = -model.likelihood.expected_log_prob(yt, qf).mean()
        rmse = (mean - yt).pow(2).mean().sqrt()

        qf_params = model(xt_ctx)
        mean, std = qf_params[:, : yt_ctx.shape[-1]], qf_params[
            :, yt_ctx.shape[-1] :
        ].pow(0.5)
        pred_std = (std.pow(2) + model.likelihood.noise.pow(2)).pow(0.5)
        qf = torch.distributions.Normal(mean, std)
        qf_pred = torch.distributions.Normal(mean, pred_std)
        nll_ctx = -qf_pred.log_prob(yt_ctx).mean()
        exp_ll_ctx = -model.likelihood.expected_log_prob(yt_ctx, qf).mean()
        rmse_ctx = (mean - yt_ctx).pow(2).mean().sqrt()

        qf_params = model(xt_trg)
        mean, std = qf_params[:, : yt_trg.shape[-1]], qf_params[
            :, yt_trg.shape[-1] :
        ].pow(0.5)
        pred_std = (std.pow(2) + model.likelihood.noise.pow(2)).pow(0.5)
        qf = torch.distributions.Normal(mean, std)
        qf_pred = torch.distributions.Normal(mean, pred_std)
        nll_trg = -qf_pred.log_prob(yt_trg).mean()
        exp_ll_trg = -model.likelihood.expected_log_prob(yt_trg, qf).mean()
        rmse_trg = (mean - yt_trg).pow(2).mean().sqrt()

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
        "nll_ctx": nll_ctx,
        "exp_ll_ctx": exp_ll_ctx,
        "nll_trg": nll_trg,
        "exp_ll_trg": exp_ll_trg,
        "rmse": rmse,
        "rmse_ctx": rmse_ctx,
        "rmse_trg": rmse_trg,
    }


def validate_dpsgp(generator, params):
    result = {
        "elbo": [],
        "nll": [],
        "exp_ll": [],
        "kl": [],
        "pred_mean": [],
        "pred_std": [],
        "gt_mean": [],
        "gt_std": [],
        "gt_loss": [],
        "nll_ctx": [],
        "exp_ll_ctx": [],
        "nll_trg": [],
        "exp_ll_trg": [],
        "rmse": [],
        "rmse_ctx": [],
        "rmse_trg": [],
    }

    batches = []
    batch_results = []
    for batch in tqdm(generator, total=generator.num_batches, desc="Validation"):

        batch_results.append(dp_train_model.remote(batch, params))

    batch_results = [ray.get(batch_result) for batch_result in batch_results]

    for batch_result in batch_results:
        result["elbo"].append(batch_result["elbo"])
        result["nll"].append(batch_result["nll"])
        result["exp_ll"].append(batch_result["exp_ll"])
        result["kl"].append(batch_result["kl"])
        result["pred_mean"].append(batch_result["mean"])
        result["pred_std"].append(batch_result["std"])

        result["gt_mean"].append(batch_result["gt_mean"][:, :, 0])
        result["gt_std"].append(batch_result["gt_std"][:, :, 0])
        result["gt_loss"].append(batch_result["gt_loss"])

        result["nll_ctx"].append(batch_result["nll_ctx"])
        result["exp_ll_ctx"].append(batch_result["exp_ll_ctx"])
        result["nll_trg"].append(batch_result["nll_trg"])
        result["exp_ll_trg"].append(batch_result["exp_ll_trg"])

        result["rmse"].append(batch_result["rmse"])
        result["rmse_ctx"].append(batch_result["rmse_ctx"])
        result["rmse_trg"].append(batch_result["rmse_trg"])

        batches.append(batch_result["batch"])

    result["elbo"] = torch.stack(result["elbo"])
    result["nll"] = torch.stack(result["nll"])
    result["exp_ll"] = torch.stack(result["exp_ll"])
    result["kl"] = torch.stack(result["kl"])
    result["mean_elbo"] = result["elbo"].mean()
    result["std_elbo"] = result["elbo"].std()
    result["mean_nll"] = result["nll"].mean()
    result["std_nll"] = result["nll"].std()
    result["mean_exp_ll"] = result["exp_ll"].mean()
    result["std_exp_ll"] = result["exp_ll"].std()
    result["mean_kl"] = result["kl"].mean()
    result["std_kl"] = result["kl"].std()

    result["nll_ctx"] = torch.stack(result["nll_ctx"])
    result["exp_ll_ctx"] = torch.stack(result["exp_ll_ctx"])
    result["nll_trg"] = torch.stack(result["nll_trg"])
    result["exp_ll_trg"] = torch.stack(result["exp_ll_trg"])
    result["mean_nll_ctx"] = result["nll_ctx"].mean()
    result["std_nll_ctx"] = result["nll_ctx"].std()
    result["mean_exp_ll_ctx"] = result["exp_ll_ctx"].mean()
    result["std_exp_ll_ctx"] = result["exp_ll_ctx"].std()
    result["mean_nll_trg"] = result["nll_trg"].mean()
    result["std_nll_trg"] = result["nll_trg"].std()
    result["mean_exp_ll_ctx"] = result["exp_ll_ctx"].mean()
    result["std_exp_ll_ctx"] = result["exp_ll_ctx"].std()

    result["rmse"] = torch.stack(result["rmse"])
    result["rmse_ctx"] = torch.stack(result["rmse_ctx"])
    result["rmse_trg"] = torch.stack(result["rmse_trg"])

    result["gt_loss"] = tf.concat(result["gt_loss"], axis=0)
    result["epsilon"] = tf.concat([b.epsilon for b in batches], axis=0)
    result["delta"] = tf.concat([b.delta for b in batches], axis=0)

    return result, batches


def evaluation_summary(
    path: str,
    evaluation_result: Dict[str, tf.Tensor],
    batches: List[Batch],
):
    # Get batch information
    batch_info = [
        get_batch_info(batch, idx)
        for batch in batches
        for idx in range(len(batch.x_ctx))
    ]

    num_ctx = np.array(
        [batch.x_ctx.shape[1] for batch in batches for _ in range(len(batch.x_ctx))]
    )

    lengthscale = np.array(
        [
            batch.gt_pred.kernel.lengthscales.numpy()
            for batch in batches
            for _ in range(len(batch.x_ctx))
        ]
    )

    # Make dataframe
    df = pd.DataFrame(
        {
            "elbo": evaluation_result["elbo"].numpy(),
            "gt_loss": evaluation_result["gt_loss"].numpy(),
            "nll": evaluation_result["nll"].numpy(),
            "exp_ll": evaluation_result["exp_ll"].numpy(),
            "kl": evaluation_result["kl"].numpy(),
            "epsilon": evaluation_result["epsilon"].numpy(),
            "delta": evaluation_result["delta"].numpy(),
            "n": num_ctx,
            "lengthscale": lengthscale,
            "nll_ctx": evaluation_result["nll_ctx"].numpy(),
            "nll_trg": evaluation_result["nll_trg"].numpy(),
            "exp_ll_ctx": evaluation_result["exp_ll_ctx"].numpy(),
            "exp_ll_trg": evaluation_result["exp_ll_trg"].numpy(),
            "rmse": evaluation_result["rmse"].numpy(),
            "rmse_ctx": evaluation_result["rmse_ctx"].numpy(),
            "rmse_trg": evaluation_result["rmse_trg"].numpy(),
        }
    )

    # Save dataframe
    df.to_csv(f"{path}/metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--evaluation_config", type=str)
    parser.add_argument("--lengthscale", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--num_ctx", type=int)
    parser.add_argument("--debug", action="store_true")
    args, config_changes = parser.parse_known_args()

    experiment_config = OmegaConf.load(f"{args.experiment_path}/config.yml")
    evaluation_config = OmegaConf.merge(
        OmegaConf.load(args.evaluation_config),
        OmegaConf.from_cli(config_changes),
    )

    evaluation_config["generator"]["min_log10_lengthscale"] = math.log10(
        args.lengthscale
    )
    evaluation_config["generator"]["max_log10_lengthscale"] = math.log10(
        args.lengthscale
    )
    evaluation_config["generator"]["min_epsilon"] = args.epsilon
    evaluation_config["generator"]["max_epsilon"] = args.epsilon
    evaluation_config["generator"]["min_num_ctx"] = args.num_ctx
    evaluation_config["generator"]["max_num_ctx"] = args.num_ctx
    evaluation_config["params"][
        "eval_name"
    ] = f"lengthscale_{args.lengthscale}_eps_{evaluation_config['generator']['min_epsilon']}_log10delta_{evaluation_config['generator']['min_log10_delta']}_min_num_ctx_{evaluation_config['generator']['min_num_ctx']}"

    experiment = instantiate(experiment_config)
    evaluation = instantiate(evaluation_config)

    experiment_path = args.experiment_path
    eval_name = evaluation.params.eval_name

    assert evaluation.params.eval_name is not None
    if not os.path.exists(f"{experiment_path}/eval/{eval_name}"):
        os.makedirs(f"{experiment_path}/eval/{eval_name}")

    generator = evaluation.generator

    batches = [batch for batch in generator]

    study_name = experiment_path
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
    )

    params = argparse.Namespace(**study.best_params)

    plot(
        path=f"{experiment_path}/eval/{eval_name}",
        model=None,
        seed=list(evaluation.params.evaluation_seed),
        batches=batches,
        params=params,
        num_fig=5,
    )

    result, batches = validate_dpsgp(generator, params)

    evaluation_summary(
        path=f"{experiment_path}/eval/{eval_name}",
        evaluation_result=result,
        batches=batches,
    )


if __name__ == "__main__":
    main()
