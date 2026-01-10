import numpy as np
import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError
from sklearn.metrics import root_mean_squared_error


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    id_col = "Id"
    target_col = "SalePrice"

    assert id_col in answers.columns, f"Missing {id_col} column in answers"
    assert target_col in answers.columns, f"Missing {target_col} column in answers"

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Missing {id_col} column in submission")

    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Missing {target_col} column in submission")

    submission = submission.sort_values(id_col)
    answers = answers.sort_values(id_col)

    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError(f"IDs in submission do not match IDs in answers")

    return {
        "y_true": answers[target_col].to_numpy(),
        "y_pred": submission[target_col].to_numpy(),
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    rmse_input = prepare_for_metric(submission, answers)
    y_true = np.log1p(rmse_input["y_true"])
    y_pred = np.log1p(rmse_input["y_pred"])
    score = root_mean_squared_error(y_true, y_pred)
    return score
