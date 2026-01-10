from pathlib import Path

import pandas as pd
from mlebench.utils import read_csv

ID_COLUMN = "ID"
TARGET_COLUMN = "rating"


def prepare(raw: Path, public: Path, private: Path):
    train_df = read_csv(raw / "train.csv")
    test_template_df = read_csv(raw / "test.csv")
    sample_submission_df = read_csv(raw / "sample_submission.csv")

    train_columns = list(train_df.columns)
    test_columns = list(test_template_df.columns)
    sample_columns = list(sample_submission_df.columns)

    # drop users with less than 5 ratings
    user_counts = train_df["user_id"].value_counts()
    train_df = train_df[train_df["user_id"].map(user_counts) >= 5].copy()

    # 2 records per user for test and all other for train
    train_df_sorted = train_df.sort_values(["user_id"])

    new_test = train_df_sorted.groupby("user_id", group_keys=False).head(2)
    new_train = train_df_sorted.drop(index=new_test.index)

    # now clean up row labels
    new_test = new_test.reset_index(drop=True)
    new_train = new_train.reset_index(drop=True)

    # create ID column
    new_test.insert(0, "ID", range(100000, 100000 + len(new_test)))
    answers_df = pd.DataFrame(
        {sample_columns[0]: new_test["ID"], sample_columns[1]: new_test["rating"]}
    )

    # fill test ratings with Null
    new_test["rating"] = None

    # save private files
    answers_df.to_csv(private / "answers.csv", index=False)
    sample_submission_df = pd.DataFrame(
        {ID_COLUMN: answers_df[ID_COLUMN], TARGET_COLUMN: 0.0}
    )

    # save public files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(public / "test.csv", index=False)
    sample_submission_df.to_csv(public / "sample_submission.csv", index=False)
