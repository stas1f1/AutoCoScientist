from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CUSTOMER_ID_COLUMN = "customer_ID"
DATE_COLUMN = "S_2"
TARGET_COLUMN = "target"
CHUNK_SIZE = 1_000_000


def _summarise_statements(csv_path: Path) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=[CUSTOMER_ID_COLUMN, DATE_COLUMN],
        parse_dates=[DATE_COLUMN],
        chunksize=CHUNK_SIZE,
    ):
        chunk_summary = chunk.groupby(CUSTOMER_ID_COLUMN).agg(
            statement_count=(DATE_COLUMN, "count"),
            last_date=(DATE_COLUMN, "max"),
        )
        summaries.append(chunk_summary)

    if not summaries:
        raise ValueError(f"No rows loaded from {csv_path}")

    combined = (
        pd.concat(summaries)
        .groupby(level=0)
        .agg(
            statement_count=("statement_count", "sum"),
            last_date=("last_date", "max"),
        )
    )
    combined.index.name = CUSTOMER_ID_COLUMN
    return combined.reset_index()


def _build_stats_dict(
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    train_labels: pd.DataFrame,
) -> dict:
    stats: dict[str, object] = {}

    stats["train_customer_count"] = int(len(train_summary))
    stats["test_customer_count"] = int(len(test_summary))

    stats["train_statement_count_distribution"] = {
        int(k): int(v)
        for k, v in train_summary["statement_count"].value_counts().sort_index().items()
    }
    stats["test_statement_count_distribution"] = {
        int(k): int(v)
        for k, v in test_summary["statement_count"].value_counts().sort_index().items()
    }

    stats["train_last_date_min"] = train_summary["last_date"].min().isoformat()
    stats["train_last_date_max"] = train_summary["last_date"].max().isoformat()
    stats["test_last_date_min"] = test_summary["last_date"].min().isoformat()
    stats["test_last_date_max"] = test_summary["last_date"].max().isoformat()

    target_counts = train_labels[TARGET_COLUMN].value_counts().sort_index()
    stats["train_target_counts"] = {int(k): int(v) for k, v in target_counts.items()}
    stats["train_target_positive_rate"] = float(train_labels[TARGET_COLUMN].mean())

    stats["train_statement_count_mean"] = float(train_summary["statement_count"].mean())
    stats["test_statement_count_mean"] = float(test_summary["statement_count"].mean())

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute structural statistics for the Amex Default Prediction dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing train_data.csv, train_labels.csv, and test_data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where summary parquet files and stats.json will be written.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = _summarise_statements(raw_dir / "train_data.csv")
    test_summary = _summarise_statements(raw_dir / "test_data.csv")
    train_labels = pd.read_csv(
        raw_dir / "train_labels.csv", usecols=[CUSTOMER_ID_COLUMN, TARGET_COLUMN]
    )

    train_summary.to_parquet(output_dir / "train_customer_summary.parquet", index=False)
    test_summary.to_parquet(output_dir / "test_customer_summary.parquet", index=False)

    stats = _build_stats_dict(train_summary, test_summary, train_labels)
    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print("Wrote:")
    print(f"  - {output_dir / 'train_customer_summary.parquet'}")
    print(f"  - {output_dir / 'test_customer_summary.parquet'}")
    print(f"  - {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
