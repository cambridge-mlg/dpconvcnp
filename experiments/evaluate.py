import sys

from utils import initialize_evaluation, tee_to_file, valid_epoch
from plot import plot
from dpconvcnp.utils import to_tensor, i32


def main():

    model, gens_eval, experiment_path = initialize_evaluation()

    breakpoint()

    seed, result, batches = valid_epoch(
        seed=[0, 0],
        model=model,
        generator=gens_eval[0],
    )
    
    plot(
        path=f"{experiment_path}/eval",
        model=model,
        seed=seed,
        epoch=0,
        batches=batches,
    )

if __name__ == "__main__":
    main()
