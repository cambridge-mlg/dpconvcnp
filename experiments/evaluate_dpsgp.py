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

    elbo, model = dpsgp.utils.dp_train_model(xc, yc, params)

    with torch.no_grad():
        qf_params = model(xt)

    mean, std = qf_params[:, : yt.shape[-1]], qf_params[:, yt.shape[-1] :].pow(0.5)
    qf = torch.distributions.Normal(mean, std)
    exp_ll = model.likelihood.expected_log_prob(yt, qf).sum()

    gt_mean, gt_std, gt_log_lik = batch.gt_pred(
        x_ctx=batch.x_ctx,
        y_ctx=batch.y_ctx,
        x_trg=batch.x_trg,
        y_trg=batch.y_trg,
    )

    gt_loss = -gt_log_lik / batch.y_trg.shape[1]

    return batch, elbo, exp_ll, mean, std, gt_mean, gt_std, gt_loss


def validate_dpsgp(generator, params):
    result = {
        "kl_diag": [],
        "loss": [],
        "pred_mean": [],
        "pred_std": [],
        "gt_mean": [],
        "gt_std": [],
        "gt_loss": [],
    }

    batches = []
    batch_results = []
    for batch in tqdm(generator, total=generator.num_batches, desc="Validation"):

        batch_results.append(dp_train_model.remote(batch, params))

    batch_results = [ray.get(batch_result) for batch_result in batch_results]

    for batch_result in batch_results:
        batch, elbo, exp_ll, mean, std, gt_mean, gt_std, gt_loss = batch_result

        result["elbo"].append(elbo)
        result["exp_ll"].append(exp_ll)
        result["pred_mean"].append(mean)
        result["pred_std"].append(std)

        result["gt_mean"].append(gt_mean[:, :, 0])
        result["gt_std"].append(gt_std[:, :, 0])
        result["gt_loss"].append(gt_loss)

        batches.append(batch)

    result["elbo"] = torch.cat(result["elbo"])
    result["exp_ll"] = torch.cat(result["exp_ll"])
    result["mean_elbo"] = result["elbo"].mean()
    result["std_elbo"] = result["elbo"].std()
    result["mean_exp_ll"] = result["exp_ll"].mean()
    result["std_exp_ll"] = result["exp_ll"].std()

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
            "exp_ll": evaluation_result["exp_ll"].numpy(),
            "epsilon": evaluation_result["epsilon"].numpy(),
            "delta": evaluation_result["delta"].numpy(),
            "n": num_ctx,
            "lengthscale": lengthscale,
        }
    )

    # Save dataframe
    df.to_csv(f"{path}/metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--evaluation_config", type=str)
    parser.add_argument("--debug", action="store_true")
    args, config_changes = parser.parse_known_args()

    experiment_config = OmegaConf.load(f"{args.experiment_path}/config.yml")
    evaluation_config = OmegaConf.merge(
        OmegaConf.load(args.evaluation_config),
        OmegaConf.from_cli(config_changes),
    )
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
    )

    result, batches = validate_dpsgp(generator, params)

    evaluation_summary(
        path=f"{experiment_path}/eval/{eval_name}",
        evaluation_result=result,
        batches=batches,
    )


if __name__ == "__main__":
    main()
