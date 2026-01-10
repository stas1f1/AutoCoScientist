from pathlib import Path

import pandas as pd
from mlebench.utils import read_csv

DATE_COLUMN = "Date"
TARGET_COLUMN = "Sales"
ID_COLUMN = "Id"
SPLIT_DATE = pd.Timestamp("2015-06-25")


def prepare(raw: Path, public: Path, private: Path):
    train_df = read_csv(raw / "train.csv")
    store_df = read_csv(raw / "store.csv")
    test_template_df = read_csv(raw / "test.csv")
    sample_submission_df = read_csv(raw / "sample_submission.csv")

    train_columns = list(train_df.columns)
    test_columns = list(test_template_df.columns)
    sample_columns = list(sample_submission_df.columns)

    train_df[DATE_COLUMN] = pd.to_datetime(train_df[DATE_COLUMN])

    new_test_full = train_df[train_df[DATE_COLUMN] >= SPLIT_DATE].copy()
    new_train = train_df[train_df[DATE_COLUMN] < SPLIT_DATE].copy()

    new_train[DATE_COLUMN] = new_train[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    new_test_full[DATE_COLUMN] = new_test_full[DATE_COLUMN].dt.strftime("%Y-%m-%d")

    new_test_public = new_test_full[
        [col for col in test_columns if col != ID_COLUMN]
    ].copy()
    new_test_public.insert(0, ID_COLUMN, range(1, len(new_test_public) + 1))
    new_test_public = new_test_public[test_columns]

    # Split on test and answer
    answers_df = pd.DataFrame(
        {
            sample_columns[0]: new_test_public[ID_COLUMN],
            sample_columns[1]: new_test_full[TARGET_COLUMN],
        }
    )

    # New sample submission
    new_sample_submission_df = pd.DataFrame(
        {sample_columns[0]: new_test_public[ID_COLUMN], sample_columns[1]: 0}
    )

    # Create private files
    answers_df.to_csv(private / "answers.csv", index=False)

    # Create public files
    new_train = new_train[train_columns]
    new_train.to_csv(public / "train.csv", index=False)
    new_test_public.to_csv(public / "test.csv", index=False)
    new_sample_submission_df.to_csv(public / "sample_submission.csv", index=False)
    store_df.to_csv(public / "store.csv", index=False)
