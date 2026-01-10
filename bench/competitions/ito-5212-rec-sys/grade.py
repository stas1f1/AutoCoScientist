import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError
from sklearn.metrics import mean_absolute_error


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    id_col = "ID"
    target_col = "rating"

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


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    metric_inputs = prepare_for_metric(submission, answers)
    return mean_absolute_error(**metric_inputs)
