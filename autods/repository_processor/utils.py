"""Tool for identifying the most important files in a repository."""

import os
from collections import Counter
from pathlib import Path
from typing import Literal

from autods.utils.repo_treesitter import RepoTreeSitter

RankingStrategy = Literal["least_internal_imports", "multi_factor_scoring"]


# Statistical method
def _extract_important_api(
    repo_path: Path, strategy: RankingStrategy = "multi_factor_scoring", top_n: int = 15
) -> list[tuple[str, float]]:
    """Extract important API files from a repository using the specified ranking strategy.

    Args:
        repo_path: Path to the repository
        strategy: Ranking strategy to use ("least_internal_imports" or "multi_factor_scoring")
        top_n: Number of top files to return (), 0 to return all

    Returns:
        List of tuples (file_path, score) representing important files and their importance scores
    """
    test_example_paths = _extract_test_example_paths(repo_path)
    test_paths = test_example_paths["test"]
    example_paths = test_example_paths["example"]

    treesitter = RepoTreeSitter(str(repo_path))
    structure = treesitter.analyze_directory(str(repo_path))

    _exclusions = [".git", ".github", "__init__", "__pycache__"]
    _exclusions = _exclusions + test_paths + example_paths

    # Split the dictionary structure
    included = {}
    excluded = {}
    for key, value in structure.items():
        if any(ex in key for ex in _exclusions):
            excluded[key] = value
        else:
            included[key] = value

    # Apply the selected ranking strategy
    if strategy == "least_internal_imports":
        results = _rank_by_least_internal_imports(included, excluded)
    elif strategy == "multi_factor_scoring":
        if repo_path is None:
            raise ValueError("repo_path is required for multi_factor_scoring strategy")
        results = _rank_by_multi_factor_scoring(included, excluded, repo_path)
    else:
        raise ValueError(f"Unknown ranking strategy: {strategy}")

    if top_n > 0:
        results = results[:top_n]
    return results


# Ranking strategies


def _rank_by_least_internal_imports(
    included: dict,
    excluded: dict,
) -> list[tuple[str, float]]:
    """Rank files by least internal imports (original strategy).

    Files that are imported least within the main codebase but are present
    in tests and examples are considered more important.

    Returns:
        List of tuples (file_path, score) where score is inversely related to import count.
    """
    # Extract all file paths from the imports and count occurrences
    file_import_counts = _get_import_counts(included)

    # Sort by least imported packages (Counter.most_common() returns list of tuples)
    most_imported = file_import_counts.most_common()
    least_imported = most_imported[::-1]

    # Remove private modules and inits
    least_imported = [
        (file, count) for file, count in least_imported if not file.startswith("_")
    ]

    # Calculate max import count for normalization
    max_imports = most_imported[0][1] if most_imported else 1

    # Get all fpaths mentioned in tests and examples
    if excluded:
        excluded_import_counts = _get_import_counts(excluded)
        imported_in_tests_and_examples = list(excluded_import_counts.keys())

        important_files = [
            (file, 100.0 * (1 - (count / (max_imports + 1))))
            for file, count in least_imported
            if file in imported_in_tests_and_examples
        ]
        less_important_files = [
            (file, 100.0 * (1 - (count / (max_imports + 1))))
            for file, count in least_imported
            if file not in imported_in_tests_and_examples
        ]
        important_files = important_files + less_important_files

    else:
        important_files = [
            (file, 100.0 * (1 - (count / (max_imports + 1))))
            for file, count in most_imported
        ]

    return important_files


def _rank_by_multi_factor_scoring(
    included: dict,
    excluded: dict,
    repo_path: Path,
) -> list[tuple[str, float]]:
    """Rank files using multi-factor scoring strategy.

    Combines multiple signals including external usage, internal import ratio,
    package hierarchy, naming conventions, documentation, and code metrics.

    Returns:
        List of tuples (file_path, score) sorted by score descending.
    """
    # Get import counts
    internal_imports = _get_import_counts(included)
    test_imports: Counter[str] = Counter()
    example_imports: Counter[str] = Counter()

    if excluded:
        excluded_imports = _get_import_counts(excluded)
        # For simplicity, treat all excluded imports as test imports
        # In a more sophisticated version, we could separate test vs example
        test_imports = excluded_imports

    # Score all files
    file_scores = []
    all_files = set(included.keys())

    for file_path, file_structure in included.items():
        score = _score_file_importance_multi_factor(
            file_path=file_path,
            repo_path=repo_path,
            file_structure=file_structure,
            test_imports=test_imports,
            example_imports=example_imports,
            internal_imports=internal_imports,
            all_files=all_files,
        )
        file_scores.append((file_path, score))

    # Sort by score descending
    file_scores.sort(key=lambda x: x[1], reverse=True)

    # Apply post-filtering
    filtered_files = []
    for file_path, score in file_scores:
        filename = Path(file_path).stem

        # Remove pure utility files below threshold
        if filename in ["utils", "helpers", "helper", "util"] and score < 10:
            continue

        # Remove private modules unless highly scored
        if (
            filename.startswith("_")
            and filename not in ["__init__", "__main__"]
            and score < 50
        ):
            continue

        # Remove __init__ files
        if filename == "__init__":
            continue

        filtered_files.append((file_path, score))

    return filtered_files


# Helper methods


def _extract_test_example_paths(repo_path: Path) -> dict[str, list]:
    """Extract paths to test and example/tutorial directories in a repository.

    Searches for test and example/tutorial folders with various naming conventions
    up to 2 levels deep from the repository root.

    Args:
        repo_path: Path to the repository root directory.

    Returns:
        Dictionary with 'test' and 'example' keys containing lists of found paths.
        Returns empty lists if no directories of that category are found.
    """
    result: dict[str, list] = {"test": [], "example": []}

    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        return result

    # Common patterns for test directories
    test_patterns = [
        "test",
        "tests",
        "testing",
        "spec",
        "specs",
        "*test*",
        "*tests*",
        "*testing*",
        "*spec*",
        "*specs*",
    ]

    # Common patterns for example/tutorial directories
    example_patterns = [
        "example",
        "examples",
        "tutorial",
        "tutorials",
        "demo",
        "demos",
        "sample",
        "samples",
        "doc",
        "docs",
        "*example*",
        "*examples*",
        "*tutorial*",
        "*tutorials*",
        "*demo*",
        "*demos*",
        "*sample*",
        "*samples*",
    ]

    def matches_pattern(dirname: str, patterns: list[str]) -> bool:
        """Check if directory name matches any of the given patterns."""
        dirname_lower = dirname.lower()
        for pattern in patterns:
            if "*" in pattern:
                # Handle wildcard patterns
                pattern_part = pattern.replace("*", "").lower()
                if pattern_part in dirname_lower:
                    return True
            else:
                # Exact match
                if dirname_lower == pattern.lower():
                    return True
        return False

    # Search up to 2 levels deep
    for root, dirs, _ in os.walk(repo_path):
        # Calculate current depth relative to repo_path
        relative_path = os.path.relpath(root, repo_path)
        depth = 0 if relative_path == "." else len(relative_path.split(os.sep))

        if depth > 2:
            # Skip deeper levels by clearing dirs list
            dirs.clear()
            continue

        # Track directories to remove from further traversal
        dirs_to_remove = []

        for dirname in dirs:
            dir_path = os.path.join(root, dirname)

            # Check if it's a test directory
            if matches_pattern(dirname, test_patterns):
                result["test"].append(dir_path)
                dirs_to_remove.append(dirname)  # Don't traverse into this directory

            # Check if it's an example/tutorial directory
            elif matches_pattern(dirname, example_patterns):
                result["example"].append(dir_path)
                dirs_to_remove.append(dirname)  # Don't traverse into this directory

        # Remove matched directories from further traversal
        for dirname in dirs_to_remove:
            if dirname in dirs:
                dirs.remove(dirname)

    # Remove duplicates and sort
    result["test"] = sorted(list(set(result["test"])))
    result["example"] = sorted(list(set(result["example"])))

    return result


def _get_import_counts(parsed_structure: dict) -> Counter:
    """Get import counts for each file in the parsed structure."""
    # Extract a flat list of import paths from the included files
    all_import_paths = []
    for structure in parsed_structure.values():
        imports_dict = structure.get("imports", {})
        # imports_dict is a dictionary where values contain 'path' information
        for import_name, import_info in imports_dict.items():
            if isinstance(import_info, dict) and "path" in import_info:
                all_import_paths.append(import_info["path"])
    return Counter(all_import_paths)


def _calculate_hierarchy_score(file_path: str, repo_path: Path) -> float:
    """Calculate package hierarchy score for a file.

    Higher scores for top-level modules and entry points.
    """
    path = Path(file_path)
    relative_path = path.relative_to(repo_path) if repo_path in path.parents else path

    score = 0.0

    # Entry point scripts
    if path.stem in ["cli", "main", "__main__"]:
        score += 40

    # Check depth in package hierarchy
    parts = relative_path.parts
    depth = len(parts) - 1  # Subtract 1 for the file itself

    if depth == 0:
        score += 30  # Package root
    elif depth == 1:
        score += 10  # First level subdirectory
    elif depth == 2:
        score += 5  # Second level subdirectory

    return min(score, 100)


def _calculate_naming_score(file_path: str) -> float:
    """Calculate naming conventions score for a file.

    Generic/abstract names get higher scores, implementation details get lower scores.
    """
    filename = Path(file_path).stem
    score = 0.0

    # Generic/abstract names
    generic_names = ["base", "core", "api", "client", "interface", "abstract"]
    if any(name in filename.lower() for name in generic_names):
        score += 20

    # Implementation details (private modules)
    if filename.startswith("_") and filename not in ["__init__", "__main__"]:
        score -= 20

    # Utility files
    if filename in ["utils", "helpers", "helper", "util"]:
        score += 5

    return max(min(score, 100), 0)  # Clamp between 0 and 100


def _calculate_documentation_score(file_path: str, file_structure: dict) -> float:
    """Calculate documentation richness score for a file.

    Well-documented files with examples get higher scores.
    """
    if not file_structure:
        return 0.0

    score = 0.0

    # Count docstrings
    structure = file_structure.get("structure", [])
    docstring_count = 0
    total_items = 0

    for item in structure:
        total_items += 1
        if item.get("docstring"):
            docstring_count += 1

        # Check methods in classes
        if item["type"] == "class":
            methods = item.get("methods", [])
            for method in methods:
                total_items += 1
                if method.get("docstring"):
                    docstring_count += 1

    # Calculate documentation ratio
    if total_items > 0:
        doc_ratio = docstring_count / total_items
        score += doc_ratio * 15

    return min(score, 100)


def _calculate_code_metrics_score(file_structure: dict) -> float:
    """Calculate code metrics score for a file.

    OOP APIs and files with few dependencies get higher scores.
    """
    if not file_structure:
        return 0.0

    score = 0.0
    structure = file_structure.get("structure", [])

    class_count = sum(1 for item in structure if item["type"] == "class")
    function_count = sum(1 for item in structure if item["type"] == "function")

    # High class-to-function ratio indicates OOP API
    if function_count > 0:
        ratio = class_count / (function_count + class_count)
        if ratio > 0.5:
            score += 10
    elif class_count > 0:
        score += 10

    # Check for abstract base classes or protocols
    for item in structure:
        if item["type"] == "class":
            # Simple heuristic: check if class name contains Base, Abstract, Protocol, Interface
            class_name = item.get("name", "")
            if any(
                keyword in class_name
                for keyword in ["Base", "Abstract", "Protocol", "Interface"]
            ):
                score += 15
                break

    return min(score, 100)


def _score_file_importance_multi_factor(
    file_path: str,
    repo_path: Path,
    file_structure: dict,
    test_imports: Counter,
    example_imports: Counter,
    internal_imports: Counter,
    all_files: set,
) -> float:
    """Calculate composite importance score for a file using multi-factor scoring."""
    score = 0.0

    # 1. External Usage Score (30%)
    test_count = test_imports.get(file_path, 0)
    example_count = example_imports.get(file_path, 0)
    external_score = test_count * 10 + example_count * 15
    score += (min(external_score, 100) / 100) * 30
    # print(test_imports, example_imports)

    # 2. Internal Import Ratio (20%)
    internal_count = internal_imports.get(file_path, 0)
    external_count = test_count + example_count
    if external_count > 0:  # Only score if there's external usage
        ratio = external_count / (internal_count + 1)
        ratio_score = min(ratio * 20, 100)
        score += (ratio_score / 100) * 20

    # 3. Package Hierarchy Score (15%)
    hierarchy_score = _calculate_hierarchy_score(file_path, repo_path)
    score += (hierarchy_score / 100) * 15

    # 4. Naming Conventions Score (10%)
    naming_score = _calculate_naming_score(file_path)
    score += (naming_score / 100) * 10

    # 5. Documentation Richness Score (10%)
    doc_score = _calculate_documentation_score(file_path, file_structure)
    score += (doc_score / 100) * 10

    # 6. Code Metrics Score (10%)
    metrics_score = _calculate_code_metrics_score(file_structure)
    score += (metrics_score / 100) * 10

    # 7. Direct Import Fanout (5%)
    fanout = internal_imports.get(file_path, 0)
    fanout_score = min(fanout * 2, 100)
    score += (fanout_score / 100) * 5

    return score


def _get_repository_tree(repo_path: Path) -> str:
    """Generate a tree structure of the repository."""
    tree_lines = []

    def _add_directory(path: Path, prefix: str = "", is_last: bool = True):
        """Recursively add directory contents to tree."""
        if path.name.startswith(".") and path.name not in [".github", ".gitignore"]:
            return

        connector = "└── " if is_last else "├── "
        tree_lines.append(f"{prefix}{connector}{path.name}")

        if path.is_dir():
            try:
                children = sorted(
                    [
                        p
                        for p in path.iterdir()
                        if not p.name.startswith(".")
                        or p.name in [".github", ".gitignore"]
                    ]
                )
                extension = "    " if is_last else "│   "

                for i, child in enumerate(children):
                    is_child_last = i == len(children) - 1
                    _add_directory(child, prefix + extension, is_child_last)
            except PermissionError:
                pass

    tree_lines.append(repo_path.name)
    try:
        children = sorted(
            [
                p
                for p in repo_path.iterdir()
                if not p.name.startswith(".") or p.name in [".github", ".gitignore"]
            ]
        )
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            _add_directory(child, "", is_last)
    except PermissionError:
        tree_lines.append("Permission denied")

    return "\n".join(tree_lines)


def _get_readme_content(repo_path: Path) -> str:
    """Get README content from the repository."""
    readme_files = [
        "README.md",
        "README.txt",
        "README.rst",
        "README",
        "readme.md",
        "readme.txt",
    ]

    for readme_file in readme_files:
        readme_path = repo_path / readme_file
        if readme_path.exists():
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    return f.read()
            except (UnicodeDecodeError, PermissionError):
                continue

    return "No README file found or readable."
