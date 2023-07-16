import sys

from utils import initialize_experiment, train_epoch, valid_epoch, plot
from dpconvcnp.utils import to_tensor, i32


def main():

    experiment, path, stdout, writer = initialize_experiment()
    sys.stdout = stdout

    dpconvcnp = experiment.model
    gen_train = experiment.generators.train
    gen_valid = experiment.generators.valid
    optimizer = experiment.optimizer
    seed = to_tensor(experiment.params.training_seed, i32)
    validation_seed = to_tensor(experiment.params.training_seed, i32)

    step = 0

    for epoch in range(100):
        
        seed, step = train_epoch(
            seed=seed,
            model=dpconvcnp,
            generator=gen_train,
            optimizer=optimizer,
            writer=writer,
            step=step,
        )

        plot_seed, _, batches = valid_epoch(
            seed=validation_seed,
            model=dpconvcnp,
            generator=gen_valid,
            writer=writer,
            epoch=epoch,
        )

        plot(
            path=path,
            model=dpconvcnp,
            seed=plot_seed,
            epoch=epoch,
            batches=batches,
        )



if __name__ == "__main__":
    main()
