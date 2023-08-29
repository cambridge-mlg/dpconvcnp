import sys

from utils import initialize_evaluation, tee_to_file, valid_epoch
from plot import plot
from dpconvcnp.utils import to_tensor, i32


def main():
    model, seed, gens_eval, experiment_path = initialize_evaluation()

    for gen in gens_eval:
        seed, result, batches = valid_epoch(
            seed=[0, 0],
            model=model,
            generator=gen,
        )

        plot(
            path=f"{experiment_path}/eval",
            model=model,
            seed=[0, 0],
            epoch=0,
            batches=batches,
        )


if __name__ == "__main__":
    main()
