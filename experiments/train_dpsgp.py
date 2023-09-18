import sys

from utils import initialize_experiment, tee_to_file
from functools import partial

import torch
import optuna
import dpsgp
import ray

torch.set_default_dtype(torch.float64)


def main():

    experiment, path, log_path, writer, checkpointer = initialize_experiment()
    tee_to_file(log_path)

    gen_train = experiment.generators.train

    # Load first batch of gen_train for hyperparameter optimisation.
    xc, yc, epsilon, delta = [], [], [], []
    for batch in gen_train:
        xc.append(torch.as_tensor(batch.x_ctx[0, ...].numpy(), dtype=torch.float64))
        yc.append(
            torch.as_tensor(batch.y_ctx[0, ...].numpy(), dtype=torch.float64)
        )
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
            experiment.limits.min_batch_size,
            experiment.limits.max_batch_size,
            experiment.limits.min_inducing,
            experiment.limits.max_inducing,
            experiment.limits.min_epochs,
            experiment.limits.max_epochs,
            experiment.limits.min_lr,
            experiment.limits.max_lr,
            experiment.limits.min_max_grad_norm,
            experiment.limits.max_max_grad_norm,
            experiment.limits.min_init_lengthscale,
            experiment.limits.max_init_lengthscale,
            experiment.limits.min_init_scale,
            experiment.limits.max_init_scale,
            experiment.limits.min_init_noise,
            experiment.limits.max_init_noise,
        ),
        n_trials=experiment.params.n_trials,
    )


if __name__ == "__main__":
    main()
