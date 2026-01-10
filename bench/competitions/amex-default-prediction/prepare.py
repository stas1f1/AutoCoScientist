from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CHUNK_SIZE = 1_000_000
TEST_FRACTION = 0.25
STATEMENT_SHARE_TOLERANCE = 0.05
RANDOM_STATE = 20241029

CUSTOMER_ID_COLUMN = "customer_ID"
TARGET_COLUMN = "target"
PREDICTION_COLUMN = "prediction"

RESOURCE_DIR = Path(__file__).resolve().parent
TRAIN_SUMMARY_PATH = RESOURCE_DIR / "train_customer_summary.parquet"
TEST_SUMMARY_PATH = RESOURCE_DIR / "test_customer_summary.parquet"
DATASET_STATS_PATH = RESOURCE_DIR / "dataset_stats.json"


def _ensure_distribution_similarity(
    new_counts: pd.Series, template_counts: pd.Series
) -> None:
    template_share = template_counts / template_counts.sum()
    new_share = new_counts / new_counts.sum()
    aligned_template = template_share.reindex(new_share.index, fill_value=0.0)
    deviations = (new_share - aligned_template).abs()
    if deviations.max() > STATEMENT_SHARE_TOLERANCE:
        raise ValueError(
            "Statement count distribution differs from template beyond tolerance."
        )


def _compute_desired_statement_counts(
    available_counts: pd.Series,
    template_counts: pd.Series,
    desired_total: int,
) -> pd.Series:
    template_share = template_counts / template_counts.sum()
    template_share = template_share.reindex(available_counts.index, fill_value=0.0)

    desired_counts = (
        (template_share * desired_total)
        .round()
        .astype(int)
        .clip(upper=available_counts)
    )

    difference = desired_total - int(desired_counts.sum())
    if difference > 0:
        for sc in template_share.sort_values(ascending=False).index:
            capacity = int(available_counts[sc] - desired_counts[sc])
            if capacity <= 0:
                continue
            adjustment = min(capacity, difference)
            desired_counts[sc] += adjustment
            difference -= adjustment
            if difference == 0:
                break
    elif difference < 0:
        difference = abs(difference)
        for sc in template_share.sort_values(ascending=True).index:
            reducible = int(desired_counts[sc])
            if reducible <= 0:
                continue
            adjustment = min(reducible, difference)
            desired_counts[sc] -= adjustment
            difference -= adjustment
            if difference == 0:
                break
    if difference != 0:
        raise ValueError("Failed to reconcile desired statement counts.")
    if desired_counts.sum() == 0:
        raise ValueError("Unable to allocate any customers to test split.")

    return desired_counts


def _sample_group_customers(
    group: pd.DataFrame,
    sample_size: int,
    positive_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    positives = group.loc[group[TARGET_COLUMN] == 1, CUSTOMER_ID_COLUMN].to_numpy()
    negatives = group.loc[group[TARGET_COLUMN] == 0, CUSTOMER_ID_COLUMN].to_numpy()

    desired_pos = int(round(sample_size * positive_rate))
    desired_pos = max(min(desired_pos, positives.size), 0)
    desired_neg = sample_size - desired_pos

    if desired_neg > negatives.size:
        desired_neg = negatives.size
        desired_pos = min(sample_size - desired_neg, positives.size)
    if desired_pos > positives.size:
        desired_pos = positives.size
        desired_neg = min(sample_size - desired_pos, negatives.size)

    pos_idx = rng.choice(positives.size, size=desired_pos, replace=False)
    neg_idx = rng.choice(negatives.size, size=desired_neg, replace=False)

    selected = np.concatenate([positives[pos_idx], negatives[neg_idx]])

    if selected.size < sample_size:
        remaining_ids = group.loc[
            ~group[CUSTOMER_ID_COLUMN].isin(selected), CUSTOMER_ID_COLUMN
        ].to_numpy()
        extra_idx = rng.choice(
            remaining_ids.size, size=sample_size - selected.size, replace=False
        )
        selected = np.concatenate([selected, remaining_ids[extra_idx]])

    return selected


def _select_test_customers(
    train_summary: pd.DataFrame,
    train_labels: pd.DataFrame,
    template_counts: pd.Series,
) -> set[str]:
    available_counts = train_summary["statement_count"].value_counts().sort_index()

    desired_total = int(round(len(train_labels) * TEST_FRACTION))
    desired_counts = _compute_desired_statement_counts(
        available_counts, template_counts, desired_total
    )

    summary_with_labels = train_summary.merge(
        train_labels, on=CUSTOMER_ID_COLUMN, how="left"
    )
    if summary_with_labels[TARGET_COLUMN].isnull().any():
        raise ValueError("Missing targets for some customers in training data.")

    rng = np.random.default_rng(RANDOM_STATE)
    positive_rate = float(summary_with_labels[TARGET_COLUMN].mean())

    selected_ids: list[np.ndarray] = []
    for statement_count, target_size in desired_counts.items():
        group = summary_with_labels[
            summary_with_labels["statement_count"] == statement_count
        ]
        if target_size <= 0:
            continue
        if len(group) <= target_size:
            selected_ids.append(group[CUSTOMER_ID_COLUMN].to_numpy())
            continue
        sampled = _sample_group_customers(group, int(target_size), positive_rate, rng)
        selected_ids.append(sampled)

    concatenated = (
        np.concatenate(selected_ids) if selected_ids else np.empty(0, dtype=str)
    )
    unique_ids = pd.Index(concatenated).unique()

    if unique_ids.empty:
        raise ValueError("Failed to sample any customers for the test split.")

    if len(unique_ids) < desired_total:
        deficit = desired_total - len(unique_ids)
        remaining_pool = summary_with_labels[
            ~summary_with_labels[CUSTOMER_ID_COLUMN].isin(unique_ids)
        ]
        if len(remaining_pool) < deficit:
            raise ValueError("Insufficient customers to satisfy desired test size.")
        additional = remaining_pool[CUSTOMER_ID_COLUMN].sample(
            n=deficit, random_state=RANDOM_STATE
        )
        unique_ids = unique_ids.union(additional)

    return set(unique_ids.astype(str))


def _write_partitioned_statements(
    source_path: Path,
    train_dest: Path,
    test_dest: Path,
    test_ids: set[str],
) -> None:
    train_written = False
    test_written = False

    for chunk in pd.read_csv(source_path, chunksize=CHUNK_SIZE):
        mask = chunk[CUSTOMER_ID_COLUMN].isin(test_ids)

        test_chunk = chunk[mask]
        if not test_chunk.empty:
            test_chunk.to_csv(
                test_dest,
                mode="w" if not test_written else "a",
                index=False,
                header=not test_written,
            )
            test_written = True

        train_chunk = chunk[~mask]
        if not train_chunk.empty:
            train_chunk.to_csv(
                train_dest,
                mode="w" if not train_written else "a",
                index=False,
                header=not train_written,
            )
            train_written = True

    if not train_written or not test_written:
        raise ValueError("One of the output splits ended up empty after partitioning.")


def _write_sample_submission(destination: Path, customer_ids: Iterable[str]) -> None:
    df = pd.DataFrame(
        {
            CUSTOMER_ID_COLUMN: sorted(customer_ids),
            PREDICTION_COLUMN: 0,
        }
    )
    df.to_csv(destination, index=False)


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw_train_data = raw / "train_data.csv"
    raw_train_labels = raw / "train_labels.csv"
    raw_sample_submission = raw / "sample_submission.csv"

    if not (
        TRAIN_SUMMARY_PATH.exists()
        and TEST_SUMMARY_PATH.exists()
        and DATASET_STATS_PATH.exists()
    ):
        raise FileNotFoundError(
            "Missing dataset summaries. Run build_dataset_stats.py to generate "
            "train_customer_summary.parquet, test_customer_summary.parquet, and dataset_stats.json."
        )

    train_summary = pd.read_parquet(TRAIN_SUMMARY_PATH)
    template_summary = pd.read_parquet(TEST_SUMMARY_PATH)

    with DATASET_STATS_PATH.open("r", encoding="utf-8") as handle:
        dataset_stats = json.load(handle)

    train_labels = pd.read_csv(raw_train_labels)

    observed_train_counts = train_summary["statement_count"].value_counts().sort_index()
    recorded_train_counts = {
        int(k): int(v)
        for k, v in dataset_stats["train_statement_count_distribution"].items()
    }
    if observed_train_counts.to_dict() != recorded_train_counts:
        raise ValueError(
            "Observed train statement count distribution differs from recorded stats."
        )

    recorded_positive_rate = float(dataset_stats["train_target_positive_rate"])
    actual_positive_rate = float(train_labels[TARGET_COLUMN].mean())
    if not np.isclose(actual_positive_rate, recorded_positive_rate, rtol=0, atol=1e-6):
        raise ValueError("Observed target positive rate differs from recorded stats.")

    template_counts = pd.Series(
        {
            int(k): int(v)
            for k, v in dataset_stats["test_statement_count_distribution"].items()
        }
    ).sort_index()

    test_ids = _select_test_customers(train_summary, train_labels, template_counts)
    train_ids = set(train_labels[CUSTOMER_ID_COLUMN]) - test_ids

    if not train_ids:
        raise ValueError("Training customer set empty after split.")

    new_test_summary = train_summary[train_summary[CUSTOMER_ID_COLUMN].isin(test_ids)]
    new_counts = new_test_summary["statement_count"].value_counts()
    _ensure_distribution_similarity(new_counts, template_counts)

    public_train_data = public / "train_data.csv"
    public_test_data = public / "test_data.csv"
    _write_partitioned_statements(
        raw_train_data,
        public_train_data,
        public_test_data,
        test_ids,
    )

    train_labels_df = train_labels[
        train_labels[CUSTOMER_ID_COLUMN].isin(train_ids)
    ].copy()
    test_labels_df = train_labels[
        train_labels[CUSTOMER_ID_COLUMN].isin(test_ids)
    ].copy()

    if train_labels_df.empty or test_labels_df.empty:
        raise ValueError("Label split produced an empty partition.")

    train_labels_df.to_csv(public / "train_labels.csv", index=False)

    _write_sample_submission(public / "sample_submission.csv", test_ids)

    sample_submission_template = pd.read_csv(
        raw_sample_submission, usecols=[CUSTOMER_ID_COLUMN]
    )
    if sample_submission_template[CUSTOMER_ID_COLUMN].duplicated().any():
        raise ValueError(
            "Raw sample submission contains duplicate customer identifiers."
        )

    answers_df = (
        test_labels_df[[CUSTOMER_ID_COLUMN, TARGET_COLUMN]]
        .rename(columns={TARGET_COLUMN: PREDICTION_COLUMN})
        .sort_values(CUSTOMER_ID_COLUMN)
        .reset_index(drop=True)
    )
    answers_df.to_csv(private / "answers.csv", index=False)

    public_test_ids = pd.read_csv(
        public / "test_data.csv", usecols=[CUSTOMER_ID_COLUMN]
    )
    if public_test_ids[CUSTOMER_ID_COLUMN].nunique() != len(test_ids):
        raise ValueError("Test data customer count mismatch after writing splits.")

    train_ids_df = pd.read_csv(public / "train_data.csv", usecols=[CUSTOMER_ID_COLUMN])
    if set(train_ids_df[CUSTOMER_ID_COLUMN]) != train_ids:
        raise ValueError("Training data customer IDs do not match expected set.")

    template_counts_observed = (
        template_summary["statement_count"].value_counts().sort_index()
    )
    if template_counts_observed.to_dict() != template_counts.to_dict():
        raise ValueError("Template summary distribution differs from recorded stats.")
