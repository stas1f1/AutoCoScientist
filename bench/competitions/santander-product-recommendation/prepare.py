from pathlib import Path

import pandas as pd
from mlebench.utils import extract, read_csv

EVAL_MONTH = pd.Timestamp("2016-05-28")


def _ensure_csv(raw: Path, archive_name: str) -> Path:
    archive_path = raw / archive_name
    csv_path = raw / archive_name.replace(".zip", "")
    if csv_path.exists():
        return csv_path
    if archive_path.exists():
        extract(archive_path, raw)
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected `{csv_path.name}` in `{raw}`")
    return csv_path


def prepare(raw: Path, public: Path, private: Path):
    train_path = _ensure_csv(raw, "train_ver2.csv.zip")
    test_path = _ensure_csv(raw, "test_ver2.csv.zip")
    sample_submission_path = _ensure_csv(raw, "sample_submission.csv.zip")

    train_df = read_csv(train_path)
    test_df = read_csv(test_path)

    product_cols = list(set(train_df.columns) - set(test_df.columns))

    new_train_df = train_df.copy()
    new_test_df = train_df.copy()

    new_train_df["fecha_dato"] = pd.to_datetime(new_train_df["fecha_dato"])
    new_test_df["fecha_dato"] = pd.to_datetime(new_test_df["fecha_dato"])

    new_train_df = new_train_df[new_train_df["fecha_dato"] < EVAL_MONTH].copy()
    new_test_df = new_test_df[new_test_df["fecha_dato"] == EVAL_MONTH].copy()

    # Split on test and answer
    for column in product_cols:
        new_test_df[column] = (
            pd.to_numeric(new_test_df[column], errors="coerce").fillna(0).astype("int8")
        )

    added_products_series = new_test_df[product_cols].apply(
        lambda row: " ".join(
            [col for col, value in zip(product_cols, row) if value > 0]
        ),
        axis=1,
    )
    answers_df = new_test_df[["ncodpers"]].copy()
    answers_df["added_products"] = added_products_series

    new_test_df = new_test_df.drop(product_cols, axis=1)

    if new_test_df.empty:
        raise ValueError(f"No rows found for evaluation month `{EVAL_MONTH}`.")

    # New sample submission
    new_sample_submission_df = new_test_df[["ncodpers"]].copy()
    new_sample_submission_df["added_products"] = "ind_tjcr_fin_ult1"

    # Create private files
    answers_df.to_csv(private / "answers.csv", index=False)

    # Create public files
    new_train_df.to_csv(public / "train.csv", index=False)
    new_test_df.to_csv(public / "test.csv", index=False)
    new_sample_submission_df.to_csv(public / "sample_submission.csv", index=False)
