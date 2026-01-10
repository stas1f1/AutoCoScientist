import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from mlebench.registry import Competition
from mlebench.utils import authenticate_kaggle_api, extract

logger = logging.getLogger(__name__)


def ensure_leaderboard_exists(competition: Competition, force: bool = False) -> Path:
    """
    Ensures the leaderboard for a given competition exists in the competition's
    directory, returning the path to it.
    If `force` is True, the leaderboard is downloaded using the Kaggle API.
    If `force` is `false`, if the leaderboard does not exist, an error is raised.
    """
    download_dir = competition.leaderboard.parent
    leaderboard_path = competition.leaderboard
    if not force:
        if leaderboard_path.exists():
            return leaderboard_path
        else:
            raise FileNotFoundError(
                f"Leaderboard not found locally for competition `{competition.id}`. Please flag this to the developers."
            )
    api = authenticate_kaggle_api()
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        response_tuple = api.competition_download_leaderboard_with_http_info(
            id=competition.id, _preload_content=False
        )
    except Exception as exc:  # pragma: no cover - upstream request failure
        raise RuntimeError(
            f"Failed to download leaderboard archive for competition `{competition.id}`."
        ) from exc

    http_response = (
        response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple
    )
    archive_bytes = getattr(http_response, "data", None)
    if not archive_bytes:
        archive_bytes = http_response.read()
    if not archive_bytes:
        raise RuntimeError(
            f"Downloaded leaderboard archive for competition `{competition.id}` was empty."
        )

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        archive_path = tmpdir_path / "leaderboard.zip"
        archive_path.write_bytes(archive_bytes)
        extract(archive_path, tmpdir_path, recursive=False)

        leaderboard_candidates = sorted(tmpdir_path.glob("**/*.csv"))
        if leaderboard_candidates:
            leaderboard_member = leaderboard_candidates[0]
            leaderboard_df = pd.read_csv(leaderboard_member)
            pd.DataFrame(
                {
                    "scoreNullable": leaderboard_df["Score"],
                    "teamId": leaderboard_df["TeamId"],
                    "hasTeamName": leaderboard_df["TeamName"].notna(),
                    "submissionDate": leaderboard_df["LastSubmissionDate"],
                    "score": leaderboard_df["Score"],
                    "hasScore": leaderboard_df["Score"].notna(),
                }
            ).to_csv(leaderboard_path, index=False)
            logger.info(
                f"Downloaded leaderboard for competition `{competition.id}` to `{download_dir.relative_to(Path.cwd()) / 'leaderboard.csv'}`."
            )
        else:
            raise RuntimeError(
                f"Failed to download leaderboard for competition `{competition.id}`."
            )
    return leaderboard_path
