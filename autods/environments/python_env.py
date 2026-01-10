from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Minimal set of libraries required for the data science workflow tools to function.
MINIMAL_DS_PACKAGES: tuple[str, ...] = (
    # "numpy<2.0.0",
    "pandas",
    # "scipy<1.13",
    "ipykernel",
    # "lightautoml[all]==0.4.1",
    # "py-boost==0.5.2",
    # "tsururu==1.1.0",
    # "replay-rec[all]==0.20.0",
    # "pytorch-lifestream==0.6.0",
)


@dataclass(slots=True)
class PythonVirtualEnvironment:
    """Metadata about the managed Python virtual environment."""

    venv_path: Path
    python_path: Path
    packages: tuple[str, ...]
    created: bool
    env_vars: dict[str, str]

    @property
    def bin_path(self) -> Path:
        return _bin_path(self.venv_path)


def ensure_virtualenv(
    project_path: Path, packages: Sequence[str] | None = None
) -> PythonVirtualEnvironment:
    """Ensure a virtual environment exists and contains minimal DS packages."""

    project_path = project_path.resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    venv_path = project_path / ".venv"
    bin_path = _bin_path(venv_path)
    python_path = _python_path(bin_path)

    created = False
    if not python_path.exists():
        logger.info("Creating project virtualenv at %s", venv_path)
        _run([sys.executable, "-m", "venv", str(venv_path)])
        created = True

    packages_tuple = tuple(dict.fromkeys(packages or MINIMAL_DS_PACKAGES))

    if created or _needs_bootstrap(venv_path, packages_tuple):
        logger.info(
            "Bootstrapping project virtualenv with minimal packages: %s",
            ", ".join(packages_tuple),
        )
        _run(
            [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
            retries=2,
        )
        if packages_tuple:
            for package in packages_tuple:
                _run(
                    [
                        str(python_path),
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        package,
                    ],
                    retries=5,
                )
        _write_bootstrap_manifest(venv_path, packages_tuple)

    env_vars = _activated_env(os.environ.copy(), venv_path, bin_path)
    _apply_process_env(env_vars)

    return PythonVirtualEnvironment(
        venv_path=venv_path,
        python_path=python_path,
        packages=packages_tuple,
        created=created,
        env_vars=env_vars,
    )


def _bin_path(venv_path: Path) -> Path:
    return venv_path / ("Scripts" if os.name == "nt" else "bin")


def _python_path(bin_path: Path) -> Path:
    return bin_path / ("python.exe" if os.name == "nt" else "python")


def _manifest_path(venv_path: Path) -> Path:
    return venv_path / ".autods_bootstrap"


def _needs_bootstrap(venv_path: Path, packages: tuple[str, ...]) -> bool:
    manifest = _manifest_path(venv_path)
    try:
        contents = manifest.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.debug("Failed to read bootstrap manifest %s: %s", manifest, exc)
        return True
    return tuple(contents) != packages


def _write_bootstrap_manifest(venv_path: Path, packages: tuple[str, ...]) -> None:
    manifest = _manifest_path(venv_path)
    try:
        manifest.write_text("\n".join(packages), encoding="utf-8")
    except OSError as exc:
        logger.debug("Failed to write bootstrap manifest %s: %s", manifest, exc)


def _activated_env(
    base_env: dict[str, str], venv_path: Path, bin_path: Path
) -> dict[str, str]:
    env = dict(base_env)
    env["VIRTUAL_ENV"] = str(venv_path)
    current_path = env.get("PATH", "")
    bin_str = str(bin_path)
    path_parts = current_path.split(os.pathsep) if current_path else []
    if not path_parts or path_parts[0] != bin_str:
        env["PATH"] = bin_str + os.pathsep + current_path if current_path else bin_str
    env.pop("PYTHONHOME", None)
    return env


def _apply_process_env(env_vars: dict[str, str]) -> None:
    os.environ.update(env_vars)


def _run(cmd: list[str], *, retries: int = 0, retry_delay: float = 1.0) -> None:
    attempts = 0
    while True:
        logger.debug("Running command: %s", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines: list[str] = []
        if process.stdout is not None:
            for line in process.stdout:
                output_lines.append(line)
                print(line, end="")
            process.stdout.close()

        return_code = process.wait()
        combined_output = "".join(output_lines)

        if return_code == 0:
            if combined_output:
                logger.debug("Command output: %s", combined_output.strip())
            return

        stdout = combined_output
        stderr = ""
        if retries > attempts and _is_ssl_error(stdout, stderr):
            wait_seconds = retry_delay * (2**attempts)
            attempts += 1
            logger.warning(
                "Command failed due to SSL error (attempt %s/%s). Retrying in %.1f s.",
                attempts,
                retries,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        truncated_stdout = stdout.strip()[-1000:] if stdout else ""
        truncated_stderr = stderr.strip()[-1000:] if stderr else ""
        message = f"Command failed with exit code {return_code}: {' '.join(cmd)}"
        if truncated_stdout:
            message += f"\nstdout: {truncated_stdout}"
        if truncated_stderr:
            message += f"\nstderr: {truncated_stderr}"
        raise RuntimeError(message)


def _is_ssl_error(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    ssl_markers = (
        "sslerror",
        "[ssl:",
        "_ssl.c",
        "unknown error (_ssl",
        "certificate verify failed",
    )
    return any(marker in combined for marker in ssl_markers)
