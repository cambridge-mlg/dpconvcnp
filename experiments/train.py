import sys

from utils import initialize_experiment, train_epoch, valid_epoch, tee_to_file
from plot import plot
from dpconvcnp.utils import to_tensor, i32


def main():

    experiment, path, log_path, writer, checkpointer = initialize_experiment()
    tee_to_file(log_path)

    dpconvcnp = experiment.model
    gen_train = experiment.generators.train
    gen_valid = experiment.generators.valid
    optimizer = experiment.optimizer
    seed = to_tensor(experiment.params.training_seed, i32)
    validation_seed = to_tensor(experiment.params.training_seed, i32)
    epochs = experiment.params.epochs

    step = 0

    for epoch in range(epochs):
        
        seed, step = train_epoch(
            seed=seed,
            model=dpconvcnp,
            generator=gen_train,
            optimizer=optimizer,
            writer=writer,
            step=step,
        )

        plot_seed, valid_result, batches = valid_epoch(
            seed=validation_seed,
            model=dpconvcnp,
            generator=gen_valid,
            writer=writer,
            epoch=epoch,
        )

        checkpointer.update_best_and_last_checkpoints(
            model=dpconvcnp,
            valid_result=valid_result,
        )

        plot(
            path=path,
            model=dpconvcnp,
            seed=plot_seed,
            epoch=epoch,
            batches=batches,
            plot_options=experiment.params.plot_options,
        )



if __name__ == "__main__":
    main()
