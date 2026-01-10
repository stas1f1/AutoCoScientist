"""Submission validation utilities."""

import ast
from pathlib import Path

REQUIRED_AUTOML_LIBRARIES = ["tsururu", "replay", "plts", "lightautoml", "py_boost"]


def validate_automl_imports(file_path: str) -> tuple[bool, str]:
    """
    Validate that a Python file imports at least one required AutoML library.

    Args:
        file_path: Path to the Python file to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passed, False otherwise
        - error_message: Empty string if valid, error description if invalid
    """
    # Check if file exists
    path = Path(file_path)
    if not path.exists():
        return False, f"Code file does not exist: {file_path}"

    if not path.is_file():
        return False, f"Code path is not a file: {file_path}"

    if path.suffix not in [".py", ".ipynb"]:
        return (
            False,
            f"Code file must be a Python file (.py) or Jupyter notebook (.ipynb): {file_path}",
        )

    # Read and parse the file
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read code file: {e}"

    # For Jupyter notebooks, extract code cells
    if path.suffix == ".ipynb":
        import json

        try:
            notebook = json.loads(content)
            cells = notebook.get("cells", [])
            code_cells = [
                cell.get("source", [])
                for cell in cells
                if cell.get("cell_type") == "code"
            ]
            # Join all code cells
            content = "\n".join(
                "".join(source) if isinstance(source, list) else source
                for source in code_cells
            )
        except Exception as e:
            return False, f"Failed to parse Jupyter notebook: {e}"

    # Parse Python code and extract imports
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return False, f"Failed to parse Python code: {e}"

    # Extract all imported module names
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get the top-level module name
                module_name = alias.name.split(".")[0]
                imported_modules.add(module_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Get the top-level module name
                module_name = node.module.split(".")[0]
                imported_modules.add(module_name)

    # Check if any required library is imported
    # Normalize library names by removing underscores and hyphens (case-sensitive)
    def normalize_name(name: str) -> str:
        return name.replace("_", "").replace("-", "")

    normalized_imported = {normalize_name(mod) for mod in imported_modules}
    normalized_required = {
        normalize_name(lib): lib for lib in REQUIRED_AUTOML_LIBRARIES
    }

    found_libraries = normalized_imported.intersection(normalized_required.keys())

    if not found_libraries:
        libraries_str = ", ".join(REQUIRED_AUTOML_LIBRARIES)
        return (
            False,
            f"Code file must import at least one of the required AutoML libraries: {libraries_str}\n"
            f"Found imports: {', '.join(sorted(imported_modules)) if imported_modules else 'none'}",
        )

    return True, ""
