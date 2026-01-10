import numpy as np
import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    id_col = "Id"
    target_col = "Sales"

    assert id_col in answers.columns, f"{id_col} not in answers"
    assert target_col in answers.columns, f"{target_col} not in answers"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"{id_col} not in submission")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"{target_col} not in submission")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("submission and answers have different lengths")

    submission = submission.sort_values(id_col).reset_index(drop=True)
    answers = answers.sort_values(id_col).reset_index(drop=True)

    if not submission[id_col].equals(answers[id_col]):
        raise InvalidSubmissionError("Submission IDs do not match answers")

    try:
        y_pred = submission[target_col].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise InvalidSubmissionError(
            "Submission contains non-numeric predictions"
        ) from exc

    try:
        y_true = answers[target_col].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise InvalidSubmissionError("Answers contain non-numeric targets") from exc

    return {"y_true": y_true, "y_pred": y_pred}


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        raise InvalidSubmissionError("All target values are zero; RMSPE undefined")

    relative_errors = ((y_pred[mask] - y_true[mask]) / y_true[mask]) ** 2
    return float(np.sqrt(np.mean(relative_errors)))


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    metric_inputs = prepare_for_metric(submission, answers)
    return rmspe(**metric_inputs)
