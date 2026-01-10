"""Extract usage examples from Jupyter notebooks.

This module provides functionality to extract self-contained usage examples
from Jupyter notebooks, tracking import-initialization-usage patterns for
library classes and functions.
"""

import ast
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class NotebookCell:
    """Represents a single cell in a Jupyter notebook."""

    cell_type: str  # "code", "markdown"
    index: int
    source: str
    outputs: List[str] = field(default_factory=list)
    execution_count: Optional[int] = None


@dataclass
class VariableTrack:
    """Tracks the usage of a variable through notebook cells."""

    api_path: str  # Fully qualified API path (e.g., "module.Class")
    variable_name: str
    import_cell_idx: int
    init_cell_idx: int
    usage_cell_indices: Set[int] = field(default_factory=set)
    is_direct_usage: bool = False  # True if initialization coincides with usage


@dataclass
class NotebookExample:
    """Represents an extracted example from a notebook."""

    source_file: str
    api_path: str
    cells: List[NotebookCell]  # All cells in the track (import, init, usage, markdown)
    example_type: str = "notebook"
    track_info: Optional[VariableTrack] = None


class NotebookExampleExtractor:
    """Extracts usage examples from Jupyter notebooks."""

    def __init__(
        self,
        repo_path: str,
        api_elements: Set[str],
        max_output_lines: int = 10,
        max_markdown_lines: int = 10,
    ):
        """Initialize the notebook example extractor.

        Args:
            repo_path: Path to the repository root
            api_elements: Set of known API elements to track (fully qualified names)
            max_output_lines: Maximum lines of cell output to include (default: 10)
            max_markdown_lines: Maximum lines of markdown cell to include (default: 50)
        """
        self.repo_path = Path(repo_path)
        self.api_elements = api_elements
        self.max_output_lines = max_output_lines
        self.max_markdown_lines = max_markdown_lines

        # Build a mapping from simplified names to fully qualified API paths
        # This allows matching imports like "from pkg import Class" to "pkg.submodule.Class"
        self.simplified_to_qualified = self._build_simplified_mapping(api_elements)

    def _build_simplified_mapping(self, api_elements: Set[str]) -> Dict[str, Set[str]]:
        """Build a mapping from simplified names to fully qualified API paths.

        For example, if api_elements contains "pkg.submodule.Class", this will create:
        - "Class" -> {"pkg.submodule.Class"}
        - "submodule.Class" -> {"pkg.submodule.Class"}

        Args:
            api_elements: Set of fully qualified API paths

        Returns:
            Dictionary mapping simplified names to sets of possible fully qualified paths
        """
        mapping = defaultdict(set)

        for api_path in api_elements:
            parts = api_path.split(".")

            # Add all possible suffixes
            # For "pkg.submodule.Class", add:
            # - "Class" -> "pkg.submodule.Class"
            # - "submodule.Class" -> "pkg.submodule.Class"
            # - "pkg.submodule.Class" -> "pkg.submodule.Class"
            for i in range(len(parts)):
                suffix = ".".join(parts[i:])
                mapping[suffix].add(api_path)

        return dict(mapping)

    def _resolve_api_path(self, candidate_path: str) -> Optional[str]:
        """Resolve a candidate API path to a fully qualified path if possible.

        Args:
            candidate_path: Candidate API path (may be simplified)

        Returns:
            Fully qualified API path if found, None otherwise
        """
        # First check if it's already a known API element
        if candidate_path in self.api_elements:
            return candidate_path

        # Try to resolve using simplified mapping
        if candidate_path in self.simplified_to_qualified:
            matches = self.simplified_to_qualified[candidate_path]

            # If there's only one match, use it
            if len(matches) == 1:
                return next(iter(matches))

            # If there are multiple matches, prefer the shortest (most likely to be correct)
            # For example, if "Class" matches both "pkg.Class" and "pkg.utils.Class",
            # prefer "pkg.Class" as it's more likely from a direct import
            return min(matches, key=len)

        return None

    def extract_from_notebook(self, notebook_path: str) -> List[NotebookExample]:
        """Extract examples from a single Jupyter notebook.

        Args:
            notebook_path: Path to the .ipynb file

        Returns:
            List of extracted NotebookExample objects
        """
        # Load notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_data = json.load(f)

        cells = self._parse_notebook_cells(notebook_data)

        # Track imports
        imports = self._extract_imports(cells)

        # Find initialization and usage patterns
        tracks = self._track_variable_usage(cells, imports)

        # Build examples from tracks
        examples = self._build_examples(notebook_path, cells, tracks)

        return examples

    def extract_from_notebooks(
        self, notebook_paths: List[str]
    ) -> List[NotebookExample]:
        """Extract examples from multiple Jupyter notebooks.

        Args:
            notebook_paths: List of paths to .ipynb files

        Returns:
            List of all extracted NotebookExample objects
        """
        all_examples = []

        for notebook_path in notebook_paths:
            try:
                examples = self.extract_from_notebook(notebook_path)
                all_examples.extend(examples)
            except Exception as e:
                # Log error and continue with next notebook
                print(f"Error processing {notebook_path}: {e}")
                continue

        return all_examples

    def _parse_notebook_cells(
        self, notebook_data: Dict[str, Any]
    ) -> List[NotebookCell]:
        """Parse notebook cells into NotebookCell objects.

        Args:
            notebook_data: Parsed JSON notebook data

        Returns:
            List of NotebookCell objects
        """
        cells = []

        for idx, cell_data in enumerate(notebook_data.get("cells", [])):
            cell_type = cell_data.get("cell_type", "")

            # Get cell source
            source = cell_data.get("source", [])
            if isinstance(source, list):
                source = "".join(source)

            # Get cell outputs (only for code cells)
            outputs = []
            if cell_type == "code":
                for output in cell_data.get("outputs", []):
                    output_text = self._extract_output_text(output)
                    if output_text:
                        outputs.append(output_text)

            execution_count = cell_data.get("execution_count")

            cell = NotebookCell(
                cell_type=cell_type,
                index=idx,
                source=source,
                outputs=outputs,
                execution_count=execution_count,
            )
            cells.append(cell)

        return cells

    def _extract_output_text(self, output: Dict[str, Any]) -> str:
        """Extract text from a cell output.

        Args:
            output: Output dictionary from notebook cell

        Returns:
            Formatted output text, truncated to max_output_lines
        """
        output_type = output.get("output_type", "")
        text_parts = []

        if output_type == "stream":
            text = output.get("text", [])
            if isinstance(text, list):
                text = "".join(text)
            text_parts.append(text)

        elif output_type in ["execute_result", "display_data"]:
            data = output.get("data", {})
            # Prefer text/plain representation
            if "text/plain" in data:
                text = data["text/plain"]
                if isinstance(text, list):
                    text = "".join(text)
                text_parts.append(text)

        elif output_type == "error":
            # Include error information
            ename = output.get("ename", "")
            evalue = output.get("evalue", "")
            text_parts.append(f"Error: {ename}: {evalue}")

        # Combine and truncate output
        full_text = "\n".join(text_parts)
        lines = full_text.split("\n")

        if len(lines) > self.max_output_lines:
            truncated_lines = lines[: self.max_output_lines]
            truncated_lines.append(
                f"... (truncated, {len(lines) - self.max_output_lines} more lines)"
            )
            return "\n".join(truncated_lines)

        return full_text

    def _strip_magic_commands(self, source: str) -> str:
        """Strip Jupyter magic commands from source code.

        Args:
            source: Source code that may contain magic commands

        Returns:
            Source code with magic commands removed
        """
        lines = source.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Remove line magics (start with %)
            if stripped.startswith("%"):
                # Keep the rest of the line if it has code after magic
                if stripped.startswith("%%"):
                    # Cell magic - skip entire line
                    continue
                else:
                    # Line magic - try to preserve code after it
                    continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _extract_imports(self, cells: List[NotebookCell]) -> Dict[str, Dict[str, Any]]:
        """Extract import statements from notebook cells.

        Args:
            cells: List of NotebookCell objects

        Returns:
            Dictionary mapping imported names to their module info and cell index
        """
        imports = {}

        for cell in cells:
            if cell.cell_type != "code":
                continue

            try:
                # Strip magic commands before parsing
                cleaned_source = self._strip_magic_commands(cell.source)

                # Suppress SyntaxWarnings from invalid escape sequences in notebook code
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SyntaxWarning)
                    tree = ast.parse(cleaned_source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            imports[name] = {
                                "module": alias.name,
                                "cell_idx": cell.index,
                            }

                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            imports[name] = {
                                "module": module,
                                "from_import": alias.name,
                                "cell_idx": cell.index,
                            }

            except SyntaxError:
                # Skip cells with syntax errors
                continue

        return imports

    def _track_variable_usage(
        self, cells: List[NotebookCell], imports: Dict[str, Dict[str, Any]]
    ) -> List[VariableTrack]:
        """Track variable initialization and usage across cells.

        Args:
            cells: List of NotebookCell objects
            imports: Import information

        Returns:
            List of VariableTrack objects
        """
        tracks = []
        variable_to_api = {}  # Maps variable names to API paths

        for cell in cells:
            if cell.cell_type != "code":
                continue

            try:
                # Strip magic commands before parsing
                cleaned_source = self._strip_magic_commands(cell.source)

                # Suppress SyntaxWarnings from invalid escape sequences in notebook code
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SyntaxWarning)
                    tree = ast.parse(cleaned_source)

                # Look for assignments (initialization) and standalone calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        # Check if RHS is a call to an imported class/function
                        api_info = self._identify_api_call(node.value, imports)

                        if api_info and api_info["api_path"] in self.api_elements:
                            # Found initialization
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    var_name = target.id

                                    # Check if this is also a usage (method call in same expression)
                                    is_direct = self._is_direct_usage(node.value)

                                    import_cell_idx = api_info.get(
                                        "import_cell_idx", cell.index
                                    )

                                    track = VariableTrack(
                                        api_path=api_info["api_path"],
                                        variable_name=var_name,
                                        import_cell_idx=import_cell_idx,
                                        init_cell_idx=cell.index,
                                        is_direct_usage=is_direct,
                                    )

                                    if not is_direct:
                                        # Track this variable for future usage
                                        variable_to_api[var_name] = (
                                            api_info["api_path"],
                                            track,
                                        )

                                    tracks.append(track)

                    # Also track standalone function calls (not assigned to variables)
                    # This is useful for utility functions like pb_to_onnx(model, path)
                    elif isinstance(node, ast.Expr) and isinstance(
                        node.value, ast.Call
                    ):
                        api_info = self._identify_api_call(node.value, imports)

                        if api_info and api_info["api_path"] in self.api_elements:
                            # Found standalone function call
                            import_cell_idx = api_info.get(
                                "import_cell_idx", cell.index
                            )

                            # Use a generated variable name for tracking
                            var_name = f"_standalone_call_{cell.index}"

                            track = VariableTrack(
                                api_path=api_info["api_path"],
                                variable_name=var_name,
                                import_cell_idx=import_cell_idx,
                                init_cell_idx=cell.index,
                                is_direct_usage=True,  # Standalone calls are always "direct"
                            )

                            tracks.append(track)

            except SyntaxError:
                continue

        # Second pass: find variable references
        for cell in cells:
            if cell.cell_type != "code":
                continue

            for var_name, (api_path, track) in variable_to_api.items():
                # Only track usage after initialization
                if cell.index > track.init_cell_idx:
                    if self._cell_references_variable(cell.source, var_name):
                        track.usage_cell_indices.add(cell.index)

        return tracks

    def _identify_api_call(
        self, node: ast.AST, imports: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Identify if an AST node is a call to a known API element.

        Args:
            node: AST node to analyze
            imports: Import information

        Returns:
            Dictionary with api_path and import_cell_idx, or None
        """
        if not isinstance(node, ast.Call):
            return None

        # Get the function/class being called
        func = node.func

        # Handle direct name (e.g., ClassName())
        if isinstance(func, ast.Name):
            name = func.id
            if name in imports:
                import_info = imports[name]
                module = import_info.get("module", "")
                from_import = import_info.get("from_import", name)

                # Build candidate API path
                if module:
                    candidate_path = f"{module}.{from_import}"
                else:
                    candidate_path = from_import

                # Try to resolve to fully qualified path
                resolved_path = self._resolve_api_path(candidate_path)

                if resolved_path:
                    return {
                        "api_path": resolved_path,
                        "import_cell_idx": import_info.get("cell_idx", 0),
                    }

                # If the full candidate didn't resolve, try just the imported name
                # This handles cases like: from py_boost import GradientBoosting
                # where candidate_path is "py_boost.GradientBoosting"
                # but the actual path is "py_boost.gpu.boosting.GradientBoosting"
                if from_import and from_import != candidate_path:
                    resolved_path = self._resolve_api_path(from_import)
                    if resolved_path:
                        return {
                            "api_path": resolved_path,
                            "import_cell_idx": import_info.get("cell_idx", 0),
                        }

                # If not resolved but the candidate is in api_elements, use it
                # This handles cases where the import path exactly matches
                if candidate_path in self.api_elements:
                    return {
                        "api_path": candidate_path,
                        "import_cell_idx": import_info.get("cell_idx", 0),
                    }

        # Handle attribute access (e.g., module.ClassName())
        elif isinstance(func, ast.Attribute):
            parts: list[str] = []
            current: ast.expr = func

            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.insert(0, current.id)

            # Check if first part is an import
            if parts and parts[0] in imports:
                import_info = imports[parts[0]]
                module = import_info.get("module", parts[0])

                # Build candidate API path
                candidate_path = module + "." + ".".join(parts[1:])

                # Try to resolve to fully qualified path
                resolved_path = self._resolve_api_path(candidate_path)

                if resolved_path:
                    return {
                        "api_path": resolved_path,
                        "import_cell_idx": import_info.get("cell_idx", 0),
                    }

                # If not resolved but candidate is in api_elements, use it
                if candidate_path in self.api_elements:
                    return {
                        "api_path": candidate_path,
                        "import_cell_idx": import_info.get("cell_idx", 0),
                    }

        return None

    def _is_direct_usage(self, node: ast.AST) -> bool:
        """Check if a Call node represents direct usage (initialization + method call).

        Args:
            node: AST node (should be ast.Call)

        Returns:
            True if this is a direct usage pattern (e.g., ClassName().method())
        """
        if not isinstance(node, ast.Call):
            return False

        # Check if the call itself is an attribute access (method call)
        if isinstance(node.func, ast.Attribute):
            # This is already a method call
            return True

        return False

    def _cell_references_variable(self, source: str, var_name: str) -> bool:
        """Check if a cell's source code references a variable.

        Args:
            source: Cell source code
            var_name: Variable name to search for

        Returns:
            True if the variable is referenced in the cell
        """
        try:
            # Suppress SyntaxWarnings from invalid escape sequences in notebook code
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                tree = ast.parse(source)

            for node in ast.walk(tree):
                # Check for name references
                if isinstance(node, ast.Name) and node.id == var_name:
                    return True

                # Check for attribute access (var_name.method())
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == var_name:
                        return True

        except SyntaxError:
            # Fallback to regex-based search
            pattern = r"\b" + re.escape(var_name) + r"\b"
            return bool(re.search(pattern, source))

        return False

    def _build_examples(
        self, notebook_path: str, cells: List[NotebookCell], tracks: List[VariableTrack]
    ) -> List[NotebookExample]:
        """Build NotebookExample objects from variable tracks.

        Args:
            notebook_path: Path to the notebook file
            cells: List of all NotebookCell objects
            tracks: List of VariableTrack objects

        Returns:
            List of NotebookExample objects
        """
        examples = []

        for track in tracks:
            # Determine which cells to include
            if track.is_direct_usage:
                # Direct usage: only need import and init cells
                cell_indices = {track.import_cell_idx, track.init_cell_idx}
            else:
                # Include import, init, and all usage cells
                cell_indices = {track.import_cell_idx, track.init_cell_idx}
                cell_indices.update(track.usage_cell_indices)

            # Add neighboring markdown cells
            cell_indices = self._expand_with_markdown(cells, cell_indices)

            # Sort indices to maintain order
            sorted_indices = sorted(cell_indices)

            # Collect the cells
            example_cells = [cells[idx] for idx in sorted_indices]

            example = NotebookExample(
                source_file=notebook_path,
                api_path=track.api_path,
                cells=example_cells,
                track_info=track,
            )

            examples.append(example)

        return examples

    def _expand_with_markdown(
        self, cells: List[NotebookCell], cell_indices: Set[int]
    ) -> Set[int]:
        """Expand cell indices to include neighboring markdown cells.

        Args:
            cells: List of all NotebookCell objects
            cell_indices: Set of cell indices to expand

        Returns:
            Expanded set of cell indices including markdown cells
        """
        expanded = set(cell_indices)

        for idx in list(cell_indices):
            # Add preceding markdown cells
            prev_idx = idx - 1
            while prev_idx >= 0 and cells[prev_idx].cell_type == "markdown":
                expanded.add(prev_idx)
                prev_idx -= 1

            # Add following markdown cells (up to next code cell)
            next_idx = idx + 1
            while next_idx < len(cells) and cells[next_idx].cell_type == "markdown":
                expanded.add(next_idx)
                next_idx += 1
                # Only add one markdown after code cell to avoid too much expansion
                break

        return expanded

    def format_example(
        self, example: NotebookExample, include_headers: bool = True
    ) -> str:
        """Format a NotebookExample as a string for inclusion in documentation.

        Args:
            example: NotebookExample to format
            include_headers: Whether to include header comments (default: True)

        Returns:
            Formatted string representation
        """
        lines = []

        # Add header (optional)
        if include_headers:
            lines.append(f"# From: {example.source_file}")
            if example.track_info:
                lines.append(f"# Variable: {example.track_info.variable_name}")
            lines.append("")

        # Add cells
        for cell in example.cells:
            if cell.cell_type == "markdown":
                # Add markdown as comments (with truncation)
                markdown_lines = cell.source.split("\n")

                if len(markdown_lines) > self.max_markdown_lines:
                    # Truncate markdown
                    for line in markdown_lines[: self.max_markdown_lines]:
                        lines.append(f"# {line}")
                    lines.append(
                        f"# ... (markdown truncated, {len(markdown_lines) - self.max_markdown_lines} more lines)"
                    )
                else:
                    for line in markdown_lines:
                        lines.append(f"# {line}")
                lines.append("")

            elif cell.cell_type == "code":
                # Add code
                lines.append(cell.source)

                # Add output if present
                if cell.outputs:
                    lines.append("")
                    lines.append("# Output:")
                    for output in cell.outputs:
                        for line in output.split("\n"):
                            lines.append(f"# {line}")
                    lines.append("")

        return "\n".join(lines)


def extract_notebook_examples_from_repository(
    repo_path: str,
    api_elements: Set[str],
    notebook_paths: Optional[List[str]] = None,
    max_output_lines: int = 5,
    max_markdown_lines: int = 5,
) -> List[NotebookExample]:
    """Convenience function to extract examples from notebooks in a repository.

    Args:
        repo_path: Path to the repository
        api_elements: Set of known API elements to track
        notebook_paths: Optional list of specific notebook paths.
                       If None, will search for all .ipynb files in repo.
        max_output_lines: Maximum lines of cell output to include
        max_markdown_lines: Maximum lines of markdown cell to include

    Returns:
        List of NotebookExample objects
    """
    extractor = NotebookExampleExtractor(
        repo_path, api_elements, max_output_lines, max_markdown_lines
    )

    # Find notebooks if not provided
    if notebook_paths is None:
        repo_path_obj = Path(repo_path)
        notebook_paths = [str(p) for p in repo_path_obj.rglob("*.ipynb")]

    return extractor.extract_from_notebooks(notebook_paths)
