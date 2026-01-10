from __future__ import annotations

from pathlib import Path

import pandas as pd
from mlebench.utils import read_csv

META_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FORECAST_HORIZON = 28
TRAIN_END_DAY = 1913
DAY_PREFIX = "d_"


def _day_number(column: str) -> int:
    if not column.startswith(DAY_PREFIX):
        msg = f"Unexpected column name: {column}"
        raise ValueError(msg)
    return int(column[len(DAY_PREFIX) :])


def _validate_target_columns(target_columns: list[str]) -> None:
    if len(target_columns) != FORECAST_HORIZON:
        raise ValueError(
            f"Expected {FORECAST_HORIZON} target columns, found {len(target_columns)}."
        )

    expected = list(range(TRAIN_END_DAY + 1, TRAIN_END_DAY + FORECAST_HORIZON + 1))
    actual = [_day_number(col) for col in target_columns]
    if actual != expected:
        raise ValueError(f"Unexpected target day range: {actual!r}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    eval_path = raw / "sales_train_evaluation.csv"
    sample_path = raw / "sample_submission.csv"
    calendar_path = raw / "calendar.csv"
    sell_prices_path = raw / "sell_prices.csv"

    evaluation_df = read_csv(eval_path)
    sample_submission_df = read_csv(sample_path)
    calendar_df = read_csv(calendar_path)
    sell_prices_df = read_csv(sell_prices_path)

    sample_eval_ids = sample_submission_df[
        sample_submission_df["id"].str.endswith("_evaluation")
    ]["id"].to_list()
    if len(sample_eval_ids) != len(evaluation_df):
        raise ValueError(
            "Sample submission evaluation rows do not match training size."
        )

    evaluation_df = evaluation_df.set_index("id").loc[sample_eval_ids].reset_index()

    day_columns = [col for col in evaluation_df.columns if col.startswith(DAY_PREFIX)]
    if not day_columns:
        raise ValueError("No day columns found in evaluation dataframe.")

    train_columns = [col for col in day_columns if _day_number(col) <= TRAIN_END_DAY]
    target_columns = [col for col in day_columns if _day_number(col) > TRAIN_END_DAY]

    _validate_target_columns(target_columns)

    new_train = evaluation_df[META_COLUMNS + train_columns].copy()
    if new_train.isnull().any().any():
        raise ValueError("Generated training dataframe contains missing values.")

    answers_df = evaluation_df[["id"] + target_columns].copy()
    if answers_df.isnull().any().any():
        raise ValueError("Generated answers dataframe contains missing values.")

    f_columns = [f"F{idx}" for idx in range(1, FORECAST_HORIZON + 1)]
    answers_df.columns = ["id"] + f_columns

    sample_submission = pd.DataFrame(
        {
            "id": sample_eval_ids,
            **{column: 0 for column in f_columns},
        }
    )

    new_train.to_csv(public / "sales_train_evaluation.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    calendar_df.to_csv(public / "calendar.csv", index=False)
    sell_prices_df.to_csv(public / "sell_prices.csv", index=False)

    answers_df.to_csv(private / "answers.csv", index=False)
