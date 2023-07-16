from typing import Tuple, Dict
import argparse
import os
import yaml
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from datetime import datetime
import git

import tensorflow as tf
import tensorflow_probability as tfp
from tensorboard.summary import Writer
from tqdm import tqdm

from dpconvcnp.data.data import Batch
from dpconvcnp.random import Seed
from dpconvcnp.data.data import DataGenerator

tfd = tfp.distributions


@tf.function(experimental_relax_shapes=True) 
def train_step(
    seed: Seed,
    model: tf.Module,
    x_ctx: tf.Tensor,
    y_ctx: tf.Tensor,
    x_trg: tf.Tensor,
    y_trg: tf.Tensor,
    epsilon: tf.Tensor,
    delta: tf.Tensor,
    optimizer: tf.optimizers.Optimizer,
) -> Tuple[Seed, tf.Tensor]:
    """Perform a single training step, returning the updateed seed and
    loss, i.e. the negative log likelihood.

    Arguments:
        seed: seed to use in the model loss.
        model: model to train.
        batch: batch of data to use in the training step.
        optimizer: optimizer to use in the training step.

    Returns:
        seed: updated seed.
        loss: negative log likelihood.
    """


    with tf.GradientTape() as tape:
        seed, loss, _, _ = model.loss(
            seed=seed,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_trg=x_trg,
            y_trg=y_trg,
            epsilon=epsilon,
            delta=delta,
        )
        loss = tf.reduce_mean(loss) / y_trg.shape[1]

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return seed, loss


def valid_step(
    seed: Seed,
    model: tf.Module,
    generator: DataGenerator,
) -> Tuple[Seed, Dict[str, tf.Tensor]]:

    result = {
        "kl_diag": [],
        "loss": [],
        "pred_mean": [],
        "pred_std": [],
        "gt_mean": [],
        "gt_std": [],
    }

    for batch in tqdm(generator, total=generator.num_batches, desc="Validation:"):
        seed, loss, mean, std = model.loss(
            seed=seed,
            x_ctx=batch.x_ctx,
            y_ctx=batch.y_ctx,
            x_trg=batch.x_trg,
            y_trg=batch.y_trg,
            epsilon=batch.epsilon,
            delta=batch.delta,
        )

        gt_mean, gt_std, _ = batch.gt_pred(
            x_ctx=batch.x_ctx,
            y_ctx=batch.y_ctx,
            x_trg=batch.x_trg,
            y_trg=batch.y_trg,
        )

        result["loss"].append(tf.reduce_mean(loss) / batch.y_trg.shape[1])
        result["pred_mean"].append(mean[:, :, 0])
        result["pred_std"].append(std[:, :, 0])

        result["gt_mean"].append(gt_mean[:, :, 0])
        result["gt_std"].append(gt_std[:, :, 0])

        result["kl_diag"].append(
            tf.reduce_mean(
                gauss_gauss_kl_diag(
                    mean_1=gt_mean,
                    std_1=gt_std,
                    mean_2=mean,
                    std_2=std,
                )
            )
        )

    result = {key: tf.reduce_mean(value) for key, value in result.items()}

    return seed, result
    

def gauss_gauss_kl_diag(
    mean_1: tf.Tensor,
    std_1: tf.Tensor,
    mean_2: tf.Tensor,
    std_2: tf.Tensor,
) -> tf.Tensor:
    """Compute the KL divergence between two diagonal Gaussians.

    Arguments:
        mean_1: mean of first Gaussian.
        std_1: standard deviation of first Gaussian.
        mean_2: mean of second Gaussian.
        std_2: standard deviation of second Gaussian.

    Returns:
        kl: KL divergence between the two Gaussians.
    """

    dist_1 = tfd.Normal(loc=mean_1, scale=std_1)
    dist_2 = tfd.Normal(loc=mean_2, scale=std_2)

    return tfd.kl_divergence(dist_1, dist_2)


def initialize_experiment() -> Tuple[DictConfig, str, Writer]:
    """Initialise experiment by parsing the config file, checking that the
    repo is clean, creating a path for the experiment, and creating a
    writer for tensorboard.

    Returns:
        experiment: experiment config object.
        path: path to experiment.
        writer: tensorboard writer.
    """

    # Make argument parser with just the config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Create a repo object and check if local repo is clean
    repo = git.Repo(search_parent_directories=True)

    # Check that the repo is clean
    assert args.debug or not repo.is_dirty(), (
        "Repo is dirty, please commit changes."
    )
    assert args.debug or not has_commits_ahead(repo), (
        "Repo has commits ahead, please push changes."
    )

    # Initialise experiment, make path and writer
    config = OmegaConf.load(args.config) 
    experiment = instantiate(config)
    path = make_experiment_path(experiment)
    writer = Writer(f"{path}/logs")

    # Write config to file together with commit hash
    with open(f"{path}/config.yml", "w") as file:
        hash = get_current_commit_hash(repo) if not args.debug else None
        config = OmegaConf.to_container(config)
        config.update({"commit": hash})
        yaml.dump(config, file, indent=4, sort_keys=False)

    return experiment, path, writer


def has_commits_ahead(repo: git.Repo) -> bool:
    """Check if there are commits ahead in the local current branch.
    
    Arguments:
        repo: git repo object.

    Returns:
        has_commits_ahead: True if there are commits ahead, False otherwise.
    """
    if repo.head.is_detached:
        assert not repo.is_dirty(), "Repo is dirty, please commit changes."
        return False

    else:
        current_branch = repo.active_branch.name
        commits = list(repo.iter_commits(f"origin/{current_branch}..{current_branch}"))
        return len(commits) > 0


def get_current_commit_hash(repo: git.Repo) -> str:
    """Get the current commit hash of the local repo.

    Arguments:
        repo: git repo object.

    Returns:
        commit_hash: current commit hash.
    """
    if repo.head.is_detached:
        return repo.commit(repo.head.object).hexsha

    else:
        return repo.head.commit.hexsha


def make_experiment_path(experiment: DictConfig) -> str:
    """Parse initialised experiment config and make up a path
    for the experiment, and create it if it doesn't exist,
    otherwise raise an error. Finally return the path.

    Arguments:
        config: config object.

    Returns:
        experiment_path: path to the experiment.
    """
    
    path = os.path.join(
        experiment.misc.results_root,
        experiment.misc.experiment_name or datetime.now().strftime("%m-%d-%H-%M-%S"),
    )

    if not os.path.exists(path):
        print(f"Making path for experiment results: {path}.")
        os.makedirs(path)

    else:
        raise ValueError(f"Path {path} already exists.")

    return path
