from pathlib import Path

import numpy as np
from mlebench.utils import read_csv
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    old_train = read_csv(raw / "train.csv")

    np_rng = np.random.default_rng(0)

    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    new_test_without_labels = new_test.drop(columns=["SalePrice"])

    # Generate more realistic sample submission based on training data distribution
    sample_submission = new_test[["Id", "SalePrice"]].copy()

    # Use log-normal distribution for more realistic house prices
    # Based on typical house price ranges in the dataset
    # Mean around $180k, std around $80k, with realistic min/max bounds
    lognormal_prices = np_rng.lognormal(
        mean=12.0, sigma=0.4, size=len(sample_submission)
    )

    # Scale and bound to realistic house price range
    # Most houses: $100k - $400k, with some outliers up to $700k
    scaled_prices = 100000 + (lognormal_prices - lognormal_prices.min()) * 300000 / (
        lognormal_prices.max() - lognormal_prices.min()
    )
    scaled_prices = np.clip(scaled_prices, 80000, 750000)

    # Add some realistic variation
    sample_submission["SalePrice"] = scaled_prices + np_rng.normal(
        0, 10000, len(scaled_prices)
    )

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # checks
    assert len(new_train) + len(new_test) == len(old_train), (
        "Train and test length should sum to the original train length"
    )
    assert len(sample_submission) == len(new_test), (
        "Sample submission should have the same length as the test set"
    )

    assert new_train.columns.tolist() == old_train.columns.tolist(), (
        "Old and new train columns should match"
    )
    assert (
        new_test_without_labels.columns.tolist() == new_train.columns.tolist()[:-1]
    ), "Public test columns should match train columns, minus the target column"
    assert new_test.columns.tolist() == new_train.columns.tolist(), (
        "Private test columns should match train columns"
    )
    assert sample_submission.columns.tolist() == [
        "Id",
        "SalePrice",
    ], "Sample submission columns should be Id, SalePrice"

    assert set(new_train["Id"]).isdisjoint(set(new_test["Id"])), (
        "Train and test ids should not overlap"
    )
