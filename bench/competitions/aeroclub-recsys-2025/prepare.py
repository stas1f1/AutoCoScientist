from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATE_COLUMN = "requestDate"
ID_COLUMN = "Id"
GROUP_COLUMN = "ranker_id"
TARGET_COLUMN = "selected"
GROUP_FILTER_THRESHOLD = 10
GROUP_SIZE_TOLERANCE = 0.05


def _build_group_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        train_df.groupby(GROUP_COLUMN)
        .agg(group_size=(ID_COLUMN, "size"), last_date=(DATE_COLUMN, "max"))
        .sort_values(["last_date", GROUP_COLUMN])
        .reset_index()
    )
    if summary["group_size"].min() <= 0:
        raise ValueError("Found empty ranker groups while building summary.")
    return summary


def _adjust_selected_mask(
    summary: pd.DataFrame, selected_mask: list[bool], difference: int
) -> None:
    if difference == 0:
        return

    candidate_indices = [
        idx
        for idx, flag in enumerate(selected_mask)
        if flag and summary.at[idx, "group_size"] <= difference
    ]
    candidate_indices.sort(key=lambda idx: summary.at[idx, "last_date"], reverse=True)

    for idx in candidate_indices:
        if summary.at[idx, "group_size"] == difference:
            selected_mask[idx] = False
            return

    size_to_indices: defaultdict[int, list[int]] = defaultdict(list)
    for idx in candidate_indices:
        size = summary.at[idx, "group_size"]
        complement = difference - size
        if complement in size_to_indices and size_to_indices[complement]:
            complement_idx = size_to_indices[complement].pop()
            selected_mask[idx] = False
            selected_mask[complement_idx] = False
            return
        size_to_indices[size].append(idx)

    limited_candidates = candidate_indices[:200]
    limited_sizes = [summary.at[idx, "group_size"] for idx in limited_candidates]
    n = len(limited_candidates)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if limited_sizes[i] + limited_sizes[j] + limited_sizes[k] == difference:
                    for idx in (
                        limited_candidates[i],
                        limited_candidates[j],
                        limited_candidates[k],
                    ):
                        selected_mask[idx] = False
                    return

    raise ValueError("Unable to adjust ranker groups to match template row count.")


def _select_test_rankers(train_df: pd.DataFrame, target_rows: int) -> set[str]:
    summary = _build_group_summary(train_df)
    if target_rows > int(summary["group_size"].sum()):
        raise ValueError("Template requires more rows than available in train data.")

    selected_mask = [False] * len(summary)
    collected_rows = 0

    for idx in range(len(summary) - 1, -1, -1):
        selected_mask[idx] = True
        collected_rows += int(summary.at[idx, "group_size"])
        if collected_rows >= target_rows:
            difference = collected_rows - target_rows
            _adjust_selected_mask(summary, selected_mask, difference)
            break
    else:
        raise ValueError("Failed to collect enough rows for test split.")

    final_rows = int(summary.loc[selected_mask, "group_size"].sum())
    if final_rows != target_rows:
        raise ValueError("Mismatch between template size and generated test split.")

    return set(summary.loc[selected_mask, GROUP_COLUMN])


def _validate_group_distributions(
    new_test: pd.DataFrame, template_test: pd.DataFrame
) -> None:
    template_sizes = template_test.groupby(GROUP_COLUMN).size()
    new_sizes = new_test.groupby(GROUP_COLUMN).size()

    if len(new_sizes) == 0 or len(template_sizes) == 0:
        raise ValueError(
            "Encountered empty group sizes during distribution validation."
        )

    share_template = (template_sizes > GROUP_FILTER_THRESHOLD).mean()
    share_new = (new_sizes > GROUP_FILTER_THRESHOLD).mean()
    if abs(share_new - share_template) > GROUP_SIZE_TOLERANCE:
        raise ValueError(
            "Group size threshold share deviates from template beyond tolerance."
        )

    avg_template = template_sizes.mean()
    avg_new = new_sizes.mean()
    if abs(avg_new - avg_template) / avg_template > GROUP_SIZE_TOLERANCE:
        raise ValueError("Average group size deviates from template beyond tolerance.")


def _assert_unique_selection_per_group(df: pd.DataFrame) -> None:
    per_group = df.groupby(GROUP_COLUMN)[TARGET_COLUMN].sum()
    if not (per_group == 1).all():
        raise ValueError("Each ranker group must contain exactly one selected option.")


def prepare(raw: Path, public: Path, private: Path) -> None:
    train_df = pd.read_parquet(raw / "train.parquet")
    test_template_df = pd.read_parquet(raw / "test.parquet")
    sample_template_df = pd.read_parquet(raw / "sample_submission.parquet")

    test_row_count = len(test_template_df)
    sample_ids = sample_template_df[ID_COLUMN].to_numpy()
    if len(sample_ids) != test_row_count:
        raise ValueError(
            "Sample submission row count does not match template test size."
        )

    test_rankers = _select_test_rankers(
        train_df[[GROUP_COLUMN, DATE_COLUMN, ID_COLUMN]], test_row_count
    )
    new_test_answers = train_df[train_df[GROUP_COLUMN].isin(test_rankers)].copy()
    new_train = train_df[~train_df[GROUP_COLUMN].isin(test_rankers)].copy()

    if len(new_test_answers) != test_row_count:
        raise ValueError("Generated test split does not match template size.")
    if new_train.empty:
        raise ValueError("Generated training split is empty.")

    new_test_answers = new_test_answers.sort_values(
        [DATE_COLUMN, GROUP_COLUMN, ID_COLUMN]
    ).reset_index(drop=True)
    new_train = new_train.sort_values(
        [DATE_COLUMN, GROUP_COLUMN, ID_COLUMN]
    ).reset_index(drop=True)

    _assert_unique_selection_per_group(new_test_answers)
    _validate_group_distributions(new_test_answers, test_template_df[[GROUP_COLUMN]])

    new_train_columns = list(train_df.columns)
    new_test_columns = list(test_template_df.columns)
    sample_columns = list(sample_template_df.columns)

    new_test_answers[ID_COLUMN] = sample_ids

    public_test = new_test_answers[new_test_columns].copy()
    new_train = new_train[new_train_columns]

    for df in (new_train, public_test, new_test_answers):
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    sample_submission = pd.DataFrame(
        {
            sample_columns[0]: sample_ids,
            sample_columns[1]: new_test_answers[GROUP_COLUMN].to_numpy(),
            sample_columns[2]: new_test_answers.groupby(GROUP_COLUMN)
            .cumcount()
            .to_numpy()
            + 1,
        }
    )

    answers_df = new_test_answers[[ID_COLUMN, GROUP_COLUMN, TARGET_COLUMN]].copy()

    # public
    new_train.to_parquet(public / "train.parquet", index=False)
    public_test.to_parquet(public / "test.parquet", index=False)
    sample_submission.to_parquet(public / "sample_submission.parquet", index=False)

    shutil.copy(raw / "jsons_raw.tar.kaggle", public / "jsons_raw.tar.kaggle")
    shutil.copy(raw / "jsons_structure.md", public / "jsons_structure.md")

    # private
    answers_df.to_csv(private / "answers.csv", index=False)
