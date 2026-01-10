#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

META_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FORECAST_HORIZON = 28
TRAIN_END_DAY = 1913
DAY_PREFIX = "d_"

LEVEL_DEFINITIONS: list[tuple[int, list[str]]] = [
    (1, ["__all__"]),
    (2, ["state_id"]),
    (3, ["store_id"]),
    (4, ["cat_id"]),
    (5, ["dept_id"]),
    (6, ["state_id", "cat_id"]),
    (7, ["state_id", "dept_id"]),
    (8, ["store_id", "cat_id"]),
    (9, ["store_id", "dept_id"]),
    (10, ["item_id"]),
    (11, ["state_id", "item_id"]),
    (12, ["store_id", "item_id"]),
]


def _day_number(column: str) -> int:
    if not column.startswith(DAY_PREFIX):
        msg = f"Unexpected column name: {column}"
        raise ValueError(msg)
    return int(column[len(DAY_PREFIX) :])


def _compute_reference_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_path = raw_dir / "sales_train_evaluation.csv"
    sample_path = raw_dir / "sample_submission.csv"
    calendar_path = raw_dir / "calendar.csv"
    sell_prices_path = raw_dir / "sell_prices.csv"

    evaluation_df = pd.read_csv(eval_path)
    sample_submission_df = pd.read_csv(sample_path)
    calendar_df = pd.read_csv(calendar_path, usecols=["d", "wm_yr_wk"])
    sell_prices_df = pd.read_csv(sell_prices_path)

    evaluation_df = evaluation_df[
        evaluation_df["id"].str.endswith("_evaluation")
    ].copy()
    if evaluation_df.empty:
        raise ValueError("Filtered evaluation dataframe is empty.")

    sample_eval_ids = sample_submission_df[
        sample_submission_df["id"].str.endswith("_evaluation")
    ]["id"].to_list()
    if len(sample_eval_ids) != len(evaluation_df):
        raise ValueError(
            "Sample submission evaluation rows do not match training size."
        )

    evaluation_df = evaluation_df.set_index("id").loc[sample_eval_ids].reset_index()

    day_columns = [
        column for column in evaluation_df.columns if column.startswith(DAY_PREFIX)
    ]
    if not day_columns:
        raise ValueError("No day columns found in evaluation dataframe.")

    train_columns = [
        column for column in day_columns if _day_number(column) <= TRAIN_END_DAY
    ]
    weight_columns = [
        column
        for column in day_columns
        if TRAIN_END_DAY - FORECAST_HORIZON + 1 <= _day_number(column) <= TRAIN_END_DAY
    ]
    if len(weight_columns) != FORECAST_HORIZON:
        raise ValueError("Unexpected number of weight columns.")

    metadata_df = evaluation_df[META_COLUMNS].copy()
    metadata_df["__all__"] = "total"

    weight_long_df = evaluation_df[META_COLUMNS + weight_columns].melt(
        id_vars=META_COLUMNS,
        value_vars=weight_columns,
        var_name="d",
        value_name="sales",
    )
    weight_long_df = weight_long_df.merge(calendar_df, on="d", how="left")
    weight_long_df = weight_long_df.merge(
        sell_prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    weight_long_df["sell_price"] = weight_long_df["sell_price"].fillna(0.0)
    weight_long_df["revenue"] = weight_long_df["sales"] * weight_long_df["sell_price"]
    base_weight_series = weight_long_df.groupby("id", sort=False)["revenue"].sum()

    metadata_with_values = metadata_df.join(evaluation_df[train_columns])
    metadata_with_values["base_weight"] = metadata_with_values["id"].map(
        base_weight_series
    )
    if metadata_with_values["base_weight"].isnull().any():
        raise ValueError("Missing revenue weights for some identifiers.")

    reference_frames: list[pd.DataFrame] = []
    for level, group_columns in LEVEL_DEFINITIONS:
        grouped_weights = metadata_with_values.groupby(
            group_columns, sort=True, dropna=False
        )["base_weight"].sum()
        grouped_values = metadata_with_values.groupby(
            group_columns, sort=True, dropna=False
        )[train_columns].sum()

        values_array = grouped_values.to_numpy(dtype=np.float64)
        if values_array.shape[1] < 2:
            raise ValueError("Insufficient history to compute RMSSE scale.")
        level_diff = np.diff(values_array, axis=1)
        level_scale = np.mean(np.square(level_diff), axis=1)

        index = grouped_values.index
        if not isinstance(index, pd.MultiIndex):
            index = pd.MultiIndex.from_arrays([index], names=group_columns)
        index_frame = index.to_frame(index=False).astype(str)
        key_strings = index_frame.apply("|".join, axis=1).to_numpy()

        reference_frames.append(
            pd.DataFrame(
                {
                    "level": level,
                    "key": key_strings,
                    "scale": level_scale,
                    "weight": grouped_weights.to_numpy(dtype=np.float64),
                }
            )
        )

    reference_df = pd.concat(reference_frames, ignore_index=True)
    if (reference_df["weight"] < 0).any():
        raise ValueError("Encountered negative weights while building reference table.")

    metadata_output = metadata_df.drop(columns="__all__")
    reference_output = reference_df.sort_values(["level", "key"]).reset_index(drop=True)
    return metadata_output, reference_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WRMSSE reference tables for the M5 Forecasting competition."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Path to the extracted Kaggle M5 dataset (containing sales_train_evaluation.csv, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where metadata.csv and wrmsse_reference.parquet will be written. Defaults to this script's directory.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df, reference_df = _compute_reference_tables(raw_dir)

    metadata_path = output_dir / "metadata.csv"
    reference_path = output_dir / "wrmsse_reference.parquet"

    metadata_df.to_csv(metadata_path, index=False)
    reference_df.to_parquet(reference_path, index=False)

    total_weight = reference_df["weight"].sum()
    print(f"Saved metadata to {metadata_path}")
    print(
        f"Saved reference table to {reference_path} (total weight={total_weight:.4f})"
    )


if __name__ == "__main__":
    main()
