from __future__ import annotations

import numpy as np
import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError

ID_COLUMN = "Id"
GROUP_COLUMN = "ranker_id"
TARGET_COLUMN = "selected"
TOP_K = 3
GROUP_SIZE_THRESHOLD = 10


def _validate_columns(submission: pd.DataFrame, answers: pd.DataFrame) -> None:
    for column in (ID_COLUMN, GROUP_COLUMN, TARGET_COLUMN):
        if column not in answers.columns:
            raise InvalidSubmissionError(f"{column} not in answers")
        if column not in submission.columns:
            raise InvalidSubmissionError(f"{column} not in submission")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers have different lengths")
    if submission[ID_COLUMN].duplicated().any():
        raise InvalidSubmissionError("Submission contains duplicate Id values")


def _coerce_integer_ranks(submission: pd.DataFrame) -> pd.Series:
    ranks = submission[TARGET_COLUMN]
    if ranks.isnull().any():
        raise InvalidSubmissionError("Submission contains missing rank values")
    if not np.issubdtype(ranks.dtype, np.integer):
        if np.isfinite(ranks).all() and np.allclose(ranks, np.floor(ranks)):
            ranks = ranks.astype(int)
        else:
            raise InvalidSubmissionError("Submission ranks must be integer values")
    if (ranks <= 0).any():
        raise InvalidSubmissionError("Submission ranks must be positive integers")
    return ranks


def _validate_group_permutations(submission: pd.DataFrame) -> None:
    group_sizes = submission.groupby(GROUP_COLUMN).size()
    min_ranks = submission.groupby(GROUP_COLUMN)[TARGET_COLUMN].min()
    max_ranks = submission.groupby(GROUP_COLUMN)[TARGET_COLUMN].max()

    if not (min_ranks == 1).all():
        raise InvalidSubmissionError("Ranks must start at 1 within each ranker group")
    if not max_ranks.equals(group_sizes):
        raise InvalidSubmissionError(
            "Ranks must end at group size within each ranker group"
        )

    duplicates = submission.groupby([GROUP_COLUMN, TARGET_COLUMN]).size()
    if not (duplicates == 1).all():
        raise InvalidSubmissionError("Each rank within a group must be unique")


def _merge_on_id(submission: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    submission_sorted = submission.sort_values(ID_COLUMN).reset_index(drop=True)
    answers_sorted = answers.sort_values(ID_COLUMN).reset_index(drop=True)

    if not submission_sorted[ID_COLUMN].equals(answers_sorted[ID_COLUMN]):
        raise InvalidSubmissionError("Submission IDs do not match answers")
    if not submission_sorted[GROUP_COLUMN].equals(answers_sorted[GROUP_COLUMN]):
        raise InvalidSubmissionError("Submission ranker_id values do not match answers")

    submission_sorted[TARGET_COLUMN] = _coerce_integer_ranks(submission_sorted)
    _validate_group_permutations(submission_sorted)
    merged = answers_sorted.copy()
    merged["predicted_rank"] = submission_sorted[TARGET_COLUMN].to_numpy()
    return merged


def _compute_hit_rate(merged: pd.DataFrame) -> float:
    hits = []
    for ranker_id, group in merged.groupby(GROUP_COLUMN):
        group_size = len(group)
        if group_size <= GROUP_SIZE_THRESHOLD:
            continue

        selected_rows = group[group[TARGET_COLUMN] == 1]
        if len(selected_rows) != 1:
            raise InvalidSubmissionError(
                "Answers must contain exactly one selected flight per ranker_id"
            )

        predicted_rank = int(selected_rows["predicted_rank"].iloc[0])
        hits.append(int(predicted_rank <= TOP_K))

    if not hits:
        raise InvalidSubmissionError("No eligible ranker groups found for evaluation")

    return float(np.mean(hits))


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    _validate_columns(submission, answers)
    merged = _merge_on_id(submission, answers)
    return _compute_hit_rate(merged)
