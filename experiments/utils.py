from typing import Tuple, Dict, List
import argparse
import os
import yaml
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from datetime import datetime
import git
import sys
import re
from io import FileIO

import tensorflow as tf
import tensorflow_probability as tfp
from tensorboard.summary import Writer
from tqdm import tqdm

from dpconvcnp.random import Seed
from dpconvcnp.data.data import DataGenerator, Batch

tfd = tfp.distributions


@tf.function(reduce_retracing=True)
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


def train_epoch(
    seed: Seed,
    model: tf.Module,
    generator: DataGenerator,
    optimizer: tf.optimizers.Optimizer,
    writer: Writer,
    step: int,
) -> Tuple[Seed, int]:
    epoch = tqdm(generator, total=generator.num_batches, desc="Training")

    for batch in epoch:
        seed, loss = train_step(
            seed=seed,
            model=model,
            x_ctx=batch.x_ctx,
            y_ctx=batch.y_ctx,
            x_trg=batch.x_trg,
            y_trg=batch.y_trg,
            epsilon=batch.epsilon,
            delta=batch.delta,
            optimizer=optimizer,
        )

        writer.add_scalar("train/loss", loss, step)
        # writer.add_scalar("lengthscale", model.dpsetconv_encoder.lengthscale, step)

        if not model.dpsetconv_encoder.amortize_y_bound:
            writer.add_scalar(
                "y_bound", model.dpsetconv_encoder.y_bound(None)[0], step
            )

        if not model.dpsetconv_encoder.amortize_w_noise:
            writer.add_scalar(
                "w_noise", model.dpsetconv_encoder.w_noise(None)[0], step
            )

        epoch.set_postfix(loss=f"{loss:.4f}")

        step = step + 1

    return seed, step


def valid_epoch(
    seed: Seed,
    model: tf.Module,
    generator: DataGenerator,
    writer: Writer,
    epoch: int,
) -> Tuple[Seed, Dict[str, tf.Tensor], List[Batch]]:
    result = {
        "kl_diag": [],
        "loss": [],
        "pred_mean": [],
        "pred_std": [],
        "gt_mean": [],
        "gt_std": [],
    }

    batches = []

    for batch in tqdm(
        generator, total=generator.num_batches, desc="Validation"
    ):
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

        batches.append(batch)

    result["loss"] = tf.reduce_mean(result["loss"])
    result["kl_diag"] = tf.reduce_mean(result["kl_diag"])

    writer.add_scalar("val/loss", result["loss"], epoch)
    writer.add_scalar("val/kl_diag", result["kl_diag"], epoch)

    return seed, result, batches


class ModelCheckpointer:
    def __init__(self, path: str):
        self.path = path
        self.best_validation_loss = float("inf")

    def update_best_and_last_checkpoints(
        self,
        model: tf.Module,
        valid_result: Dict[str, tf.Tensor],
    ) -> None:
        """Update the best and last checkpoints of the model.

        Arguments:
            model: model to save.
            valid_result: validation result dictionary.
        """

        if valid_result["loss"] < self.best_validation_loss:
            self.best_validation_loss = valid_result["loss"]
            model.save_weights(f"{self.path}/best")

        model.save_weights(f"{self.path}/last")

    def load_best_checkpoint(self, model: tf.Module) -> None:
        model.load_weights(f"{self.path}/best")

    def load_last_checkpoint(self, model: tf.Module) -> None:
        model.load_weights(f"{self.path}/last")


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


def initialize_experiment() -> (
    Tuple[DictConfig, str, str, Writer, ModelCheckpointer]
):
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
    args, config_changes = parser.parse_known_args()

    # Create a repo object and check if local repo is clean
    repo = git.Repo(search_parent_directories=True)

    # Check that the repo is clean
    assert (
        args.debug or not repo.is_dirty()
    ), "Repo is dirty, please commit changes."
    assert args.debug or not has_commits_ahead(
        repo
    ), "Repo has commits ahead, please push changes."

    # Initialise experiment, make path and writer
    OmegaConf.register_new_resolver("eval", eval)
    config = OmegaConf.load(args.config)
    config_changes = OmegaConf.from_cli(config_changes)

    config = OmegaConf.merge(config, config_changes)
    experiment = instantiate(config)
    path = make_experiment_path(experiment)
    writer = Writer(f"{path}/logs")

    # Write config to file together with commit hash
    with open(f"{path}/config.yml", "w") as file:
        hash = get_current_commit_hash(repo) if not args.debug else None
        config = OmegaConf.to_container(config)
        config.update({"commit": hash})
        yaml.dump(config, file, indent=4, sort_keys=False)

    # Set path for logging training output messages
    log_path = f"{path}/stdout.txt"

    # Create model checkpointer
    model_checkpointer = ModelCheckpointer(path=f"{path}/checkpoints")

    return experiment, path, log_path, writer, model_checkpointer


def initialize_evaluation():
    # Make argument parser with just the config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--evaluation_config", type=str)
    parser.add_argument("--debug", action="store_true")
    args, _ = parser.parse_known_args()

    # Create a repo object and check if local repo is clean
    repo = git.Repo(search_parent_directories=True)

    # Check that the repo is clean
    assert (
        args.debug or not repo.is_dirty()
    ), "Repo is dirty, please commit changes."
    assert args.debug or not has_commits_ahead(
        repo
    ), "Repo has commits ahead, please push changes."

    # Initialise experiment, make path and writer
    OmegaConf.register_new_resolver("eval", eval)
    experiment_config = OmegaConf.load(f"{args.experiment_path}/config.yml")
    evaluation_config = OmegaConf.load(args.evaluation_config)

    # Check out commit hash -- only the model is loaded using this hash
    repo.git.checkout(experiment_config.commit)

    # Create model checkpointer and load model
    checkpointer = ModelCheckpointer(
        path=f"{args.experiment_path}/checkpoints",
    )

    # Load model weights
    model = checkpointer.load_best_checkpoint(
        model=instantiate(experiment_config.model),
    )

    # Check out previous branch
    repo.git.checkout("-")

    # Load evaluation generators
    gens_eval = instantiate(evaluation_config.generators.eval)

    return model, gens_eval, args.experiment_path


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
        commits = list(
            repo.iter_commits(f"origin/{current_branch}..{current_branch}")
        )
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
        experiment.misc.experiment_name
        or datetime.now().strftime("%m-%d-%H-%M-%S"),
    )

    if not os.path.exists(path):
        print(f"Making path for experiment results: {path}.")
        os.makedirs(path)

    else:
        raise ValueError(f"Path {path} already exists.")

    return path


def tee_to_file(log_file_path: str):
    log_file = open(log_file_path, "a")

    class Logger(object):
        def __init__(self, file: FileIO):
            self.terminal = sys.stdout
            self.log_file = file

        def write(self, message: str):
            self.terminal.write(message)
            self.log_file.write(message)

        def flush(self):
            self.terminal.flush()
            self.log_file.flush()

    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)
