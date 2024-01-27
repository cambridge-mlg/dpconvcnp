from utils import initialize_evaluation, evaluation_summary, valid_epoch
from plot import plot


def main():
    (
        model,
        base_seed,
        generator,
        experiment_path,
        eval_dirname,
        eval_name,
        plot_options,
    ) = initialize_evaluation()

    seed = base_seed
    seed, result, batches = valid_epoch(
        seed=seed,
        model=model,
        generator=generator,
    )

    plot(
        path=f"{experiment_path}/{eval_dirname}/{eval_name}",
        model=model,
        seed=base_seed,
        batches=batches,
        plot_options=plot_options,
    )

    evaluation_summary(
        path=f"{experiment_path}/{eval_dirname}/{eval_name}",
        evaluation_result=result,
        batches=batches,
    )


if __name__ == "__main__":
    main()
