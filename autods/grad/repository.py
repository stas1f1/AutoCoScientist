import subprocess
from pathlib import Path
from urllib.parse import urlparse


def get_repository_id(url: str) -> str:
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise RuntimeError(
            f"The provided URL does not appear to be a GitHub URL. Must be in format `https://github.com/owner/repo`. \nGot: {url}"
        )
    owner, repo_name = path_parts[:2]
    return f"{owner.lower()}-{repo_name.lower()}"


def clone_repository(url: str, path: str | Path):
    try:
        subprocess.run(
            ["git", "clone", url, str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to clone repository from {url}: {e.stderr}. "
            f"Please provide a valid git URL or ensure the repository exists locally."
        )
