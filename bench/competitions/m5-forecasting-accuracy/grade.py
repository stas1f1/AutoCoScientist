from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError

FORECAST_HORIZON = 28
ID_COLUMN = "id"
F_COLUMNS = [f"F{idx}" for idx in range(1, FORECAST_HORIZON + 1)]
LEVEL_DEFINITIONS = {
    1: ["__all__"],
    2: ["state_id"],
    3: ["store_id"],
    4: ["cat_id"],
    5: ["dept_id"],
    6: ["state_id", "cat_id"],
    7: ["state_id", "dept_id"],
    8: ["store_id", "cat_id"],
    9: ["store_id", "dept_id"],
    10: ["item_id"],
    11: ["state_id", "item_id"],
    12: ["store_id", "item_id"],
}
EPSILON = 1e-8


def _load_metadata() -> pd.DataFrame:
    metadata_path = Path(__file__).resolve().parent / "metadata.csv"
    metadata_df = pd.read_csv(metadata_path)
    metadata_df["__all__"] = "total"
    return metadata_df


def _load_reference() -> pd.DataFrame:
    reference_path = Path(__file__).resolve().parent / "wrmsse_reference.parquet"
    reference_df = pd.read_parquet(reference_path)
    return reference_df


def _validate_submission_structure(submission: pd.DataFrame) -> None:
    missing_columns = [
        column for column in [ID_COLUMN, *F_COLUMNS] if column not in submission.columns
    ]
    if missing_columns:
        raise InvalidSubmissionError(
            f"Submission missing required columns: {', '.join(missing_columns)}"
        )


def _prepare_frames(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_submission_structure(submission)

    missing_answer_columns = [
        column for column in [ID_COLUMN, *F_COLUMNS] if column not in answers.columns
    ]
    if missing_answer_columns:
        raise InvalidSubmissionError(
            f"Answers missing required columns: {', '.join(missing_answer_columns)}"
        )

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers must have identical length."
        )

    submission = submission[[ID_COLUMN, *F_COLUMNS]].copy()
    answers = answers[[ID_COLUMN, *F_COLUMNS]].copy()

    if submission[ID_COLUMN].duplicated().any():
        raise InvalidSubmissionError("Submission contains duplicate identifiers.")
    if answers[ID_COLUMN].duplicated().any():
        raise InvalidSubmissionError("Answers contain duplicate identifiers.")

    submission.sort_values(ID_COLUMN, inplace=True, ignore_index=True)
    answers.sort_values(ID_COLUMN, inplace=True, ignore_index=True)

    if not submission[ID_COLUMN].equals(answers[ID_COLUMN]):
        raise InvalidSubmissionError("Submission identifiers do not match answers.")

    try:
        submission[F_COLUMNS] = submission[F_COLUMNS].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidSubmissionError(
            "Submission contains non-numeric forecast values."
        ) from exc

    try:
        answers[F_COLUMNS] = answers[F_COLUMNS].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidSubmissionError(
            "Answers contain non-numeric forecast values."
        ) from exc

    if submission[F_COLUMNS].isnull().any().any():
        raise InvalidSubmissionError("Submission contains missing forecast values.")

    if answers[F_COLUMNS].isnull().any().any():
        raise InvalidSubmissionError("Answers contain missing forecast values.")

    if (submission[F_COLUMNS] < 0).any().any():
        raise InvalidSubmissionError("Forecast values must be non-negative.")

    return submission, answers


def _compute_level_scores(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    reference_by_level: dict[int, pd.DataFrame],
    normalised_weights: dict[int, pd.Series],
) -> tuple[np.ndarray, np.ndarray]:
    errors: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    metadata_by_id = metadata.set_index(ID_COLUMN)
    predictions = metadata_by_id.join(predictions.set_index(ID_COLUMN), how="inner")
    targets = metadata_by_id.join(targets.set_index(ID_COLUMN), how="inner")

    if len(predictions) != len(metadata):
        raise InvalidSubmissionError("Missing metadata for some identifiers.")

    for level, group_cols in LEVEL_DEFINITIONS.items():
        grouped_pred = predictions.groupby(group_cols, sort=True)[F_COLUMNS].sum()
        grouped_true = targets.groupby(group_cols, sort=True)[F_COLUMNS].sum()

        if len(grouped_pred) != len(grouped_true):
            raise InvalidSubmissionError(
                f"Mismatched aggregation counts for level {level}"
            )

        index = grouped_pred.index
        if not isinstance(index, pd.MultiIndex):
            index = pd.MultiIndex.from_arrays([index], names=group_cols)

        key_strings = ["|".join(str(part) for part in key_tuple) for key_tuple in index]
        reference = reference_by_level.get(level)
        weight_lookup = normalised_weights.get(level)
        if reference is None or weight_lookup is None:
            raise InvalidSubmissionError(f"Reference data missing for level {level}")

        try:
            scale_values = reference.loc[key_strings, "scale"].to_numpy(
                dtype=np.float64
            )
            weight_values = weight_lookup.loc[key_strings].to_numpy(dtype=np.float64)
        except KeyError as exc:
            raise InvalidSubmissionError(
                f"Aggregated keys missing in reference for level {level}"
            ) from exc

        diff = grouped_pred.to_numpy(dtype=np.float64) - grouped_true.to_numpy(
            dtype=np.float64
        )
        mse = np.mean(np.square(diff), axis=1)
        scale = np.maximum(scale_values, EPSILON)
        rmsse = np.sqrt(mse / scale)

        errors.append(rmsse)
        weights.append(weight_values)

    return np.concatenate(errors), np.concatenate(weights)


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _prepare_frames(submission, answers)


def weighted_rmsse(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    metadata = _load_metadata()
    reference_df = _load_reference()

    total_weight = reference_df["weight"].sum()
    if total_weight <= 0:
        raise InvalidSubmissionError("Reference weights must be positive.")

    reference_by_level = {
        level: frame.set_index("key") for level, frame in reference_df.groupby("level")
    }
    normalised_weights = {
        level: frame.set_index("key")["weight"] / total_weight
        for level, frame in reference_df.groupby("level")
    }

    errors, weights = _compute_level_scores(
        predictions, targets, metadata, reference_by_level, normalised_weights
    )
    score = float(np.sum(errors * weights))
    return score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    predictions, targets = prepare_for_metric(submission, answers)
    return weighted_rmsse(predictions, targets)
