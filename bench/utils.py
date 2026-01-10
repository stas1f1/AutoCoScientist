from pathlib import Path


def get_module_dir() -> Path:
    """Returns an absolute path to the MLE-bench module."""

    path = Path(__file__).parent.resolve()

    assert path.name == "bench", (
        f"Expected the module directory to be `mlebench`, but got `{path.name}`."
    )

    return path


def get_repo_dir() -> Path:
    """Returns an absolute path to the repository directory."""

    return get_module_dir().parent
