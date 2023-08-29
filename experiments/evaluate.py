import sys

from utils import initialize_evaluation, tee_to_file
from dpconvcnp.utils import to_tensor, i32


def main():

    experiment, path, checkpointer = initialize_evaluation()

    dpconvcnp = experiment.model
    gens_eval = experiment.generators.eval




if __name__ == "__main__":
    main()
