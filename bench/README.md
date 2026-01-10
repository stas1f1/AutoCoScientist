# AutoDS-bench

## Leaderboard

---

## Benchmarking

This section describes a canonical setup for comparing scores on AutoDS-bench. We recommend the following:
- Repeat each evaluation with at least 3 seeds and report the Any Medal (%) score as the mean ± one standard error of the mean. The evaluation (task and grading) itself is deterministic, but agents/LLMs can be quite high-variance!
- Agent resources - not a strict requirement of the benchmark but please report if you stray from these defaults!
  - Runtime: 24 hours
  - Compute: 36 vCPUs with 440GB RAM and one 24GB A10 GPU
- Include a breakdown of your scores across Low, Medium, High, and All complexity [splits](experiments/splits) (see

## Setup

Some AutoDS-bench competition data is stored using [Git-LFS](https://git-lfs.com/).
Once you have downloaded and installed LFS, run:

```console
git lfs fetch --all
git lfs pull
```

You can install `bench` with pip:

```console
pip install -e .
```

### Pre-Commit Hooks (Optional)

If you're committing code, you can install the pre-commit hooks by running:

```console
pre-commit install
```

## Dataset

The AutoDS-bench dataset is a collection of Kaggle competitions which we use to evaluate the ML engineering capabilities of AI systems.

Since Kaggle does not provide the held-out test set for each competition, we
provide preparation scripts that split the publicly available training set into
a new training and test set.

For each competition, we also provide grading scripts that can be used to
evaluate the score of a submission.

We use the [Kaggle API](https://github.com/Kaggle/kaggle-api) to download the
raw datasets. Ensure that you have downloaded your Kaggle credentials
(`kaggle.json`) and placed it in the `~/.kaggle/` directory (this is the default
location where the Kaggle API looks for your credentials). To download and prepare the AutoDS-bench dataset, run the following, which will download and prepare the dataset in your system's default cache directory. Note, we've found this to take two days when running from scratch:

```console
bench prepare --all
```

To prepare the lite dataset, run:

```console
bench prepare --lite
```

Alternatively, you can prepare the dataset for a specific competition by
running:

```console
bench prepare -c <competition-id>
```

Run `bench prepare --help` to see the list of available competitions.



## Grading Submissions

Answers for competitions must be submitted in CSV format; the required format is described in each competition's description, or shown in a competition's sample submission file. You can grade multiple submissions by using the `bench grade` command. Given a JSONL file, where each line corresponds with a submission for one competition, `bench grade` will produce a grading report for each competition. The JSONL file must contain the following fields:
- `competition_id`: the ID of the competition in our dataset.
- `submission_path`: a `.csv` file with the predictions for the specified
  competition.

See more information by running `bench grade --help`.

You can also grade individual submissions using the `bench grade-sample` command. For example, to grade a submission for the Spaceship Titanic competition, you can run:

```console
bench grade-sample <PATH_TO_SUBMISSION> spaceship-titanic
```

See more information by running `bench grade-sample --help`.

## Experiments

We place the code specific to the experiments of the
benchmark in the `bench/experiments/` directory:
- For instance, our competition splits are available in `experiments/splits/`.
- For a completed set of runs from a given agent, you can use the provided
`experiments/make_submission.py` script to compile its submission for grading.
