from typing import Tuple, Any, Dict
import argparse
import os
import json
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from datetime import datetime

import tensorflow as tf
from tensorboard.summary import Writer
import matplotlib.pyplot as plt

from dpconvcnp.data.data import Batch
from dpconvcnp.random import Seed


def train_step(
    seed: Seed,
    model: tf.Module,
    batch: Batch,
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
        seed, loss = model.loss(
            seed=seed,
            x_ctx=batch.x_ctx,
            y_ctx=batch.y_ctx,
            x_trg=batch.x_trg,
            y_trg=batch.y_trg,
            epsilon=batch.epsilon,
            delta=batch.delta,
        )
        loss = tf.reduce_mean(loss) / batch.y_trg.shape[1]

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return seed, loss


def initialize_experiment() -> Tuple[DictConfig, str, Writer]:

    # Make argument parser with just the config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)

    # Parse the config argument and make OmegaConf object
    args = parser.parse_args()

    # Initialise experiment, make path and writer
    config = OmegaConf.load(args.config) 
    experiment = instantiate(config)
    path = make_experiment_path(experiment)
    writer = Writer(f"{path}/logs")

    # Write config to file
    with open(f"{path}/config.json", "w") as file:
        config = OmegaConf.to_container(config)
        json.dump(, file)

    return experiment, path, writer


def add_git_info(config: Dict[str, Any]) -> Dict[str, Any]:

    config.misc.git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    config.misc.git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()

    return config


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
