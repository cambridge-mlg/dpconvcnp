import sys

from utils import initialize_evaluation, tee_to_file
from dpconvcnp.utils import to_tensor, i32


def main():

    model, gens_eval, experiment_path = initialize_evaluation()


if __name__ == "__main__":
    main()
