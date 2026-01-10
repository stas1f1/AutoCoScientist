from __future__ import annotations

import numpy as np
import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError

CUSTOMER_ID_COLUMN = "customer_ID"
PREDICTION_COLUMN = "prediction"
ANSWER_COLUMN = "prediction"


def _validate_submission(submission: pd.DataFrame, answers: pd.DataFrame) -> None:
    for column in (CUSTOMER_ID_COLUMN, PREDICTION_COLUMN):
        if column not in submission.columns:
            raise InvalidSubmissionError(
                f"Submission missing required column: {column}"
            )

    for column in (CUSTOMER_ID_COLUMN, ANSWER_COLUMN):
        if column not in answers.columns:
            raise InvalidSubmissionError(f"Answers missing required column: {column}")

    if submission[CUSTOMER_ID_COLUMN].duplicated().any():
        raise InvalidSubmissionError(
            "Submission contains duplicated customer identifiers."
        )
    if answers[CUSTOMER_ID_COLUMN].duplicated().any():
        raise InvalidSubmissionError("Answers contain duplicated customer identifiers.")

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers must have identical length."
        )


def _prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    _validate_submission(submission, answers)

    submission_sorted = submission.sort_values(CUSTOMER_ID_COLUMN).reset_index(
        drop=True
    )
    answers_sorted = answers.sort_values(CUSTOMER_ID_COLUMN).reset_index(drop=True)

    if not submission_sorted[CUSTOMER_ID_COLUMN].equals(
        answers_sorted[CUSTOMER_ID_COLUMN]
    ):
        raise InvalidSubmissionError(
            "Submission customer identifiers do not align with answers."
        )

    try:
        predictions = submission_sorted[PREDICTION_COLUMN].astype(np.float64).to_numpy()
    except (TypeError, ValueError) as exc:
        raise InvalidSubmissionError("Submission predictions must be numeric.") from exc

    try:
        targets = answers_sorted[ANSWER_COLUMN].astype(np.float64).to_numpy()
    except (TypeError, ValueError) as exc:
        raise InvalidSubmissionError("Answer targets must be numeric.") from exc

    return targets, predictions


def _weighted_gini(
    y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray
) -> float:
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    weights_sorted = weights[order]

    cumulative_weights = np.cumsum(weights_sorted)
    total_weight = cumulative_weights[-1]
    if total_weight <= 0:
        return 0.0

    weighted_true = y_true_sorted * weights_sorted
    cumulative_true = np.cumsum(weighted_true)
    total_true = cumulative_true[-1]
    if total_true <= 0:
        return 0.0

    lorentz = cumulative_true / total_true
    random = cumulative_weights / total_weight
    gini = np.sum((lorentz - random) * weights_sorted)
    return float(gini / total_weight)


def _amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    weights = np.where(y_true == 0, 20.0, 1.0)

    order = np.argsort(-y_pred)
    y_true_desc = y_true[order]
    weights_desc = weights[order]
    cumulative_weight = np.cumsum(weights_desc)
    total_weight = weights_desc.sum()
    cutoff = 0.04 * total_weight

    positives = y_true_desc.sum()
    if positives <= 0:
        top_four = 0.0
    else:
        mask = cumulative_weight <= cutoff
        top_four = float(y_true_desc[mask].sum() / positives)

    gini = _weighted_gini(y_true, y_pred, weights)
    gini_max = _weighted_gini(y_true, y_true, weights)
    normalized_gini = gini / gini_max if gini_max > 0 else 0.0

    return 0.5 * (normalized_gini + top_four)


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> dict[str, np.ndarray]:
    y_true, y_pred = _prepare_for_metric(submission, answers)
    return {"y_true": y_true, "y_pred": y_pred}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    metric_inputs = prepare_for_metric(submission, answers)
    return _amex_metric(**metric_inputs)
