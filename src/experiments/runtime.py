import platform
import subprocess
import sys
from importlib import metadata

TRACKED_PACKAGES = (
    "langchain",
    "langchain-core",
    "langchain-together",
    "pandas",
    "yfinance",
    "vaderSentiment",
    "edgar",
    "arize-phoenix",
)


def get_git_commit_sha(repo_root="."):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def collect_runtime_metadata():
    # Start with an empty dictionary.
    packages = {}

    for package_name in TRACKED_PACKAGES:
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
