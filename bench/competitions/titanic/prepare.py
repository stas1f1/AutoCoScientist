from pathlib import Path

from mlebench.utils import read_csv
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Create random example submission
    example_submission = new_test[["PassengerId", "Survived"]].copy()
    example_submission["Survived"] = 0
    example_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    new_test.drop("Survived", axis="columns").to_csv(public / "test.csv", index=False)
