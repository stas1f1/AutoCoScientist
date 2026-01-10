import pandas as pd
from mlebench.competitions.utils import prepare_for_accuracy_metric
from sklearn.metrics import accuracy_score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission,
        answers=answers,
        target_column="Survived",
        id_column="PassengerId",
    )
    return accuracy_score(**accuracy_inputs)
