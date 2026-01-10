from pathlib import Path

import pandas as pd
from mlebench.utils import read_csv

DATE_COLUMN = "date"
TARGET_COLUMN = "sales"
ID_COLUMN = "id"


def _ensure_same_combinations(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    ref_combos = (
        reference[["store_nbr", "family"]]
        .drop_duplicates()
        .sort_values(["store_nbr", "family"])
        .reset_index(drop=True)
    )
    cand_combos = (
        candidate[["store_nbr", "family"]]
        .drop_duplicates()
        .sort_values(["store_nbr", "family"])
        .reset_index(drop=True)
    )

    if not ref_combos.equals(cand_combos):
        raise ValueError(
            "Store/family combinations in the generated test set differ from the template."
        )


def prepare(raw: Path, public: Path, private: Path) -> None:
    train_df = read_csv(raw / "train.csv")
    test_template_df = read_csv(raw / "test.csv")
    sample_submission_df = read_csv(raw / "sample_submission.csv")

    holidays_events_df = read_csv(raw / "holidays_events.csv")
    oil_df = read_csv(raw / "oil.csv")
    stores_df = read_csv(raw / "stores.csv")
    transactions_df = read_csv(raw / "transactions.csv")

    train_columns = list(train_df.columns)
    test_columns = list(test_template_df.columns)
    sample_columns = list(sample_submission_df.columns)

    train_df[DATE_COLUMN] = pd.to_datetime(train_df[DATE_COLUMN])
    test_template_df[DATE_COLUMN] = pd.to_datetime(test_template_df[DATE_COLUMN])
    transactions_df[DATE_COLUMN] = pd.to_datetime(transactions_df[DATE_COLUMN])

    test_days = test_template_df[DATE_COLUMN].nunique()
    expected_test_rows = len(test_template_df)
    min_test_date = test_template_df[DATE_COLUMN].min()

    split_start = train_df[DATE_COLUMN].max() - pd.Timedelta(days=test_days - 1)

    new_test_answers = (
        train_df[train_df[DATE_COLUMN] >= split_start]
        .sort_values([DATE_COLUMN, "store_nbr", "family", ID_COLUMN])
        .reset_index(drop=True)
    )
    new_train = (
        train_df[train_df[DATE_COLUMN] < split_start]
        .sort_values([DATE_COLUMN, "store_nbr", "family", ID_COLUMN])
        .reset_index(drop=True)
    )

    if len(new_test_answers) != expected_test_rows:
        raise ValueError("Generated test split does not match expected number of rows.")

    if new_test_answers[DATE_COLUMN].nunique() != test_days:
        raise ValueError(
            "Generated test split does not cover the expected number of days."
        )

    if new_test_answers[DATE_COLUMN].min() >= min_test_date:
        raise ValueError("Generated test occurs on or after the template test period.")

    if new_train.empty:
        raise ValueError("Generated training split is empty.")

    _ensure_same_combinations(test_template_df, new_test_answers)

    if new_test_answers.groupby(DATE_COLUMN).size().nunique() != 1:
        raise ValueError("Generated test split does not have uniform daily coverage.")

    id_values = new_test_answers[ID_COLUMN].to_numpy()
    if len(sample_submission_df) != len(id_values):
        raise ValueError("Sample submission size does not match generated test size.")

    sample_submission_df[ID_COLUMN] = id_values

    public_test = new_test_answers[test_columns].copy()

    answers_df = new_test_answers[train_columns].copy()

    new_sample_submission = pd.DataFrame(
        {
            sample_columns[0]: id_values,
            sample_columns[1]: 0.0,
        }
    )

    transactions_df = transactions_df[transactions_df[DATE_COLUMN] < split_start]

    for df in (new_train, public_test, answers_df, transactions_df):
        df[DATE_COLUMN] = df[DATE_COLUMN].dt.strftime("%Y-%m-%d")

    new_train = new_train[train_columns]
    public_test = public_test[test_columns]

    new_train.to_csv(public / "train.csv", index=False)
    public_test.to_csv(public / "test.csv", index=False)
    new_sample_submission.to_csv(public / "sample_submission.csv", index=False)

    answers_df.to_csv(private / "test.csv", index=False)

    holidays_events_df.to_csv(public / "holidays_events.csv", index=False)
    oil_df.to_csv(public / "oil.csv", index=False)
    stores_df.to_csv(public / "stores.csv", index=False)
    transactions_df.to_csv(public / "transactions.csv", index=False)
