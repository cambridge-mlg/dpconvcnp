import sys

from utils import initialize_experiment, tee_to_file
from functools import partial

import torch
import optuna
import dpsgp
import ray
import wandb

torch.set_default_dtype(torch.float64)


def main():
    experiment, config, path, log_path, _, _ = initialize_experiment()
    tee_to_file(log_path)

    gen_train = experiment.generators.train

    # Load first batch of gen_train for hyperparameter optimisation.
    xc, yc, epsilon, delta = [], [], [], []
    for batch in gen_train:
        xc.append(torch.as_tensor(batch.x_ctx[0, ...].numpy(), dtype=torch.float64))
        yc.append(torch.as_tensor(batch.y_ctx[0, ...].numpy(), dtype=torch.float64))
        epsilon.append(batch.epsilon[0].numpy())
        delta.append(batch.delta[0].numpy())

    # Create optuna study.
    study_name = path
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True,
    )

    # Create wandb callback.
    wandbc = optuna.integration.WeightsAndBiasesCallback(
        metric_name="elbo",
        wandb_kwargs={
            "project": experiment.misc.project,
            "name": experiment.misc.name,
            "config": config,
        },
    )

    sys.stdout.fileno = lambda: False
    sys.stderr.fileno = lambda: False
    ray.init()
    study.optimize(
        partial(
            dpsgp.utils.objective,
            epsilon,
            delta,
            xc,
            yc,
            experiment.limits,
            experiment.kernel,
        ),
        n_trials=experiment.params.n_trials,
        callbacks=[wandbc],
    )

    # Save best parameters.
    wandb.run.summary["best_params"] = study.best_params
    wandb.run.summary["best_value"] = study.best_value


if __name__ == "__main__":
    main()
