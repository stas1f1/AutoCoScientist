"""GitHub URL utilities for converting blob URLs to raw format."""

import re
from urllib.parse import urlparse


def is_github_blob_url(url: str) -> bool:
    """
    Check if URL is a GitHub blob URL that can be converted to raw format.

    Args:
        url: URL to check

    Returns:
        True if URL is a GitHub blob URL, False otherwise

    Examples:
        >>> is_github_blob_url("https://github.com/user/repo/blob/main/file.py")
        True
        >>> is_github_blob_url("https://github.com/user/repo/tree/main/src")
        False
    """
    parsed = urlparse(url)

    if parsed.hostname != "github.com":
        return False

    # Pattern: /user/repo/blob/branch/path/to/file
    blob_pattern = r"^/([^/]+)/([^/]+)/blob/(.+)$"
    return bool(re.match(blob_pattern, parsed.path))


def is_colab_github_url(url: str) -> bool:
    """
    Check if URL is a Google Colab URL pointing to a GitHub file.

    Colab URLs have the pattern:
    https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/{path}

    Args:
        url: URL to check

    Returns:
        True if URL is a Colab GitHub URL, False otherwise

    Examples:
        >>> is_colab_github_url("https://colab.research.google.com/github/user/repo/blob/main/notebook.ipynb")
        True
        >>> is_colab_github_url("https://github.com/user/repo/blob/main/notebook.ipynb")
        False
    """
    parsed = urlparse(url)

    if parsed.hostname != "colab.research.google.com":
        return False

    # Pattern: /github/user/repo/blob/branch/path/to/file
    colab_pattern = r"^/github/([^/]+)/([^/]+)/blob/(.+)$"
    return bool(re.match(colab_pattern, parsed.path))


def convert_colab_url_to_raw(url: str) -> str:
    """
    Convert Google Colab GitHub URLs to raw.githubusercontent.com format.

    Extracts the embedded GitHub path from Colab URLs and converts to raw format
    for direct content access.

    Args:
        url: Google Colab URL to convert

    Returns:
        Converted raw GitHub URL if input is a Colab GitHub URL, original URL otherwise

    Examples:
        Convert Colab URLs:
        >>> convert_colab_url_to_raw("https://colab.research.google.com/github/user/repo/blob/main/notebook.ipynb")
        'https://raw.githubusercontent.com/user/repo/main/notebook.ipynb'

        Keep non-Colab URLs unchanged:
        >>> convert_colab_url_to_raw("https://github.com/user/repo/blob/main/file.py")
        'https://github.com/user/repo/blob/main/file.py'
    """
    parsed = urlparse(url)

    if parsed.hostname != "colab.research.google.com":
        return url

    # Pattern: /github/user/repo/blob/branch/path/to/file
    colab_pattern = r"^/github/([^/]+)/([^/]+)/blob/(.+)$"
    match = re.match(colab_pattern, parsed.path)

    if match:
        user, repo, rest = match.groups()
        # rest = "branch/path/to/file.ipynb"
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{rest}"

        # Preserve query string and fragment if present
        if parsed.query:
            raw_url += f"?{parsed.query}"
        if parsed.fragment:
            raw_url += f"#{parsed.fragment}"

        return raw_url

    return url


def convert_github_url_to_raw(url: str) -> str:
    """
    Convert GitHub blob URLs and Colab GitHub URLs to raw.githubusercontent.com format.

    This enables direct access to file content without GitHub's HTML wrapper,
    which is better for web scraping and content extraction.

    Supports:
    - Regular GitHub blob URLs: github.com/{user}/{repo}/blob/{branch}/{path}
    - Google Colab URLs: colab.research.google.com/github/{user}/{repo}/blob/{branch}/{path}

    Args:
        url: GitHub or Colab URL to convert

    Returns:
        Converted raw URL if input is a blob or Colab URL, original URL otherwise

    Examples:
        Convert blob URLs:
        >>> convert_github_url_to_raw("https://github.com/user/repo/blob/main/README.md")
        'https://raw.githubusercontent.com/user/repo/main/README.md'

        Convert Colab URLs:
        >>> convert_github_url_to_raw("https://colab.research.google.com/github/user/repo/blob/main/notebook.ipynb")
        'https://raw.githubusercontent.com/user/repo/main/notebook.ipynb'

        Keep tree URLs unchanged:
        >>> convert_github_url_to_raw("https://github.com/user/repo/tree/main/src")
        'https://github.com/user/repo/tree/main/src'

        Keep non-GitHub URLs unchanged:
        >>> convert_github_url_to_raw("https://example.com/page")
        'https://example.com/page'

        Already raw URLs unchanged:
        >>> convert_github_url_to_raw("https://raw.githubusercontent.com/user/repo/main/file.py")
        'https://raw.githubusercontent.com/user/repo/main/file.py'
    """
    colab_converted = convert_colab_url_to_raw(url)
    if colab_converted != url:
        return colab_converted

    parsed = urlparse(url)

    if parsed.hostname != "github.com":
        return url

    # Pattern: /user/repo/blob/branch/path/to/file
    blob_pattern = r"^/([^/]+)/([^/]+)/blob/(.+)$"
    match = re.match(blob_pattern, parsed.path)

    if match:
        user, repo, rest = match.groups()
        # rest = "branch/path/to/file.py"
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{rest}"

        # Preserve query string and fragment if present
        if parsed.query:
            raw_url += f"?{parsed.query}"
        if parsed.fragment:
            raw_url += f"#{parsed.fragment}"

        return raw_url

    return url


def id_to_collection_name(repo_id: str, prefix: str = "") -> str:
    """Transforms ID in NWO format to collection name.

    Args:
        repo_id: Repository ID in NWO format: owner/repo
        prefix: Optional prefix for name

    Returns:
        Collection name without spaces or dots
    """

    repo_id = repo_id.replace(" ", "_").replace(".", "_")

    return prefix + repo_id
