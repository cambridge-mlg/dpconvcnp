from utils import initialize_evaluation, evaluation_summary, valid_epoch
from plot import plot


def main():
    model, base_seed, generator, experiment_path = initialize_evaluation()

    seed = base_seed
    seed, result, batches = valid_epoch(
        seed=seed,
        model=model,
        generator=generator,
    )

    plot(
        path=f"{experiment_path}/eval",
        model=model,
        seed=base_seed,
        batches=batches,
    )

    evaluation_summary(
        path=f"{experiment_path}/eval",
        evaluation_result=result,
        batches=batches,
    )


if __name__ == "__main__":
    main()
