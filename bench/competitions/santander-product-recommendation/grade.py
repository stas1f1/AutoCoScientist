import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.metrics import mean_average_precision_at_k


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    id_col = "ncodpers"
    target_col = "added_products"

    assert id_col in answers.columns, f"{id_col} not in answers"
    assert target_col in answers.columns, f"{target_col} not in answers"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"{id_col} not in submission")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"{target_col} not in submission")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("submission and answers have different lengths")

    submission = submission.sort_values(by=id_col).reset_index(drop=True)
    answers = answers.sort_values(by=id_col).reset_index(drop=True)

    y_true = answers[target_col].astype(str).str.split(" ").apply(set).tolist()
    y_pred = submission[target_col].astype(str).str.split(" ").tolist()

    return {"actual": y_true, "predicted": y_pred}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    prepped = prepare_for_metric(submission, answers)
    return mean_average_precision_at_k(
        actual=prepped["actual"], predicted=prepped["predicted"], k=7
    )
