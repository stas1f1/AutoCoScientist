import numpy as np
import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError

ID_COLUMN = "id"
TARGET_COLUMN = "sales"


def _validate_submission_columns(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> None:
    if ID_COLUMN not in submission.columns:
        raise InvalidSubmissionError(f"{ID_COLUMN} not in submission")
    if TARGET_COLUMN not in submission.columns:
        raise InvalidSubmissionError(f"{TARGET_COLUMN} not in submission")
    if ID_COLUMN not in answers.columns or TARGET_COLUMN not in answers.columns:
        raise InvalidSubmissionError("Answers missing required columns")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers have different lengths")


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> dict[str, np.ndarray]:
    _validate_submission_columns(submission, answers)

    submission = submission.sort_values(ID_COLUMN).reset_index(drop=True)
    answers = answers.sort_values(ID_COLUMN).reset_index(drop=True)

    if not submission[ID_COLUMN].equals(answers[ID_COLUMN]):
        raise InvalidSubmissionError("Submission IDs do not match answers")

    try:
        y_pred = submission[TARGET_COLUMN].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise InvalidSubmissionError(
            "Submission contains non-numeric predictions"
        ) from exc

    try:
        y_true = answers[TARGET_COLUMN].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise InvalidSubmissionError("Answers contain non-numeric targets") from exc

    if np.any(y_pred < 0):
        raise InvalidSubmissionError("Predicted sales must be non-negative for RMSLE")
    if np.any(y_true < 0):
        raise InvalidSubmissionError("Answers contain negative sales values")

    return {"y_true": y_true, "y_pred": y_pred}


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = np.log1p(y_pred) - np.log1p(y_true)
    return float(np.sqrt(np.mean(np.square(errors))))


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    metric_inputs = prepare_for_metric(submission, answers)
    return rmsle(**metric_inputs)
