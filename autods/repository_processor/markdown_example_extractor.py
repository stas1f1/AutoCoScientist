"""Extract usage examples from Markdown files.

This module provides functionality to extract self-contained usage examples
from Markdown documentation files (like README.md), tracking code blocks
and their associated documentation context.
"""

import ast
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class MarkdownCodeBlock:
    """Represents a Python code block from a Markdown file."""

    source_file: str
    code: str
    language: str  # e.g., "python", "py"
    start_line: int
    end_line: int
    preceding_header: Optional[str] = None
    preceding_text: Optional[str] = None


@dataclass
class MarkdownExample:
    """Represents an extracted example from a Markdown file."""

    source_file: str
    api_paths: Set[str]  # All API paths mentioned in the code block
    code: str
    header: Optional[str] = None
    description: Optional[str] = None
    example_type: str = "readme"


class MarkdownExampleExtractor:
    """Extracts usage examples from Markdown files."""

    def __init__(self, repo_path: str, api_elements: Set[str]):
        """Initialize the markdown example extractor.

        Args:
            repo_path: Path to the repository root
            api_elements: Set of known API elements to track (fully qualified names)
        """
        self.repo_path = Path(repo_path)
        self.api_elements = api_elements

        # Build a mapping from simplified names to fully qualified API paths
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
        from collections import defaultdict

        mapping = defaultdict(set)

        for api_path in api_elements:
            parts = api_path.split(".")

            # Add all possible suffixes
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

            # If there are multiple matches, prefer the shortest
            return min(matches, key=len)

        return None

    def extract_from_markdown(self, markdown_path: str) -> List[MarkdownExample]:
        """Extract examples from a single Markdown file.

        Args:
            markdown_path: Path to the .md file

        Returns:
            List of extracted MarkdownExample objects
        """
        # Read the markdown file
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract code blocks with context
        code_blocks = self._extract_code_blocks(content, markdown_path)

        # Process each code block to find API references
        examples = []
        for block in code_blocks:
            if block.language.lower() in ["python", "py", "python3"]:
                example = self._process_code_block(block)
                if example and example.api_paths:
                    examples.append(example)

        return examples

    def extract_from_markdowns(
        self, markdown_paths: List[str]
    ) -> List[MarkdownExample]:
        """Extract examples from multiple Markdown files.

        Args:
            markdown_paths: List of paths to .md files

        Returns:
            List of all extracted MarkdownExample objects
        """
        all_examples = []

        for markdown_path in markdown_paths:
            try:
                examples = self.extract_from_markdown(markdown_path)
                all_examples.extend(examples)
            except Exception as e:
                # Log error and continue with next file
                print(f"Error processing {markdown_path}: {e}")
                continue

        return all_examples

    def _extract_code_blocks(
        self, content: str, source_file: str
    ) -> List[MarkdownCodeBlock]:
        """Extract code blocks from Markdown content with their context.

        Args:
            content: Markdown file content
            source_file: Path to the source file (for reference)

        Returns:
            List of MarkdownCodeBlock objects
        """
        blocks = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for fenced code block (```language)
            fence_match = re.match(r"^```(\w+)?", line)
            if fence_match:
                language = fence_match.group(1) or "text"
                start_line = i + 1  # Line after the fence
                code_lines: list[str] = []

                # Find preceding header and text
                header, text = self._find_preceding_context(lines, i)

                # Collect code until closing fence
                i += 1
                while i < len(lines):
                    if lines[i].strip().startswith("```"):
                        # Found closing fence
                        end_line = i - 1
                        code = "\n".join(code_lines)

                        block = MarkdownCodeBlock(
                            source_file=source_file,
                            code=code,
                            language=language,
                            start_line=start_line,
                            end_line=end_line,
                            preceding_header=header,
                            preceding_text=text,
                        )
                        blocks.append(block)
                        break

                    code_lines.append(lines[i])
                    i += 1

            i += 1

        return blocks

    def _find_preceding_context(
        self, lines: List[str], code_start_index: int
    ) -> tuple[Optional[str], Optional[str]]:
        """Find the preceding header and text paragraph before a code block.

        Args:
            lines: All lines in the markdown file
            code_start_index: Index of the line where code block starts

        Returns:
            Tuple of (header, text_paragraph)
        """
        header = None
        text_lines: list[str] = []

        # Search backwards from the code block
        i = code_start_index - 1

        # First, collect text paragraph (immediately preceding the code)
        while i >= 0:
            line = lines[i].strip()

            # Skip empty lines initially
            if not line:
                i -= 1
                if text_lines:  # Stop if we already have text
                    break
                continue

            # Check for header (markdown headers: #, ##, ###, etc.)
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Found a header - this is our header
                header = header_match.group(2).strip()
                break

            # Regular text - add to paragraph
            text_lines.insert(0, line)
            i -= 1

        # If we didn't find a header yet, keep looking
        if header is None and i >= 0:
            i -= 1
            while i >= 0:
                line = lines[i].strip()

                if not line:
                    i -= 1
                    continue

                # Check for header
                header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
                if header_match:
                    header = header_match.group(2).strip()
                    break

                i -= 1

        # Join text lines
        text = " ".join(text_lines) if text_lines else None

        return header, text

    def _process_code_block(
        self, block: MarkdownCodeBlock
    ) -> Optional[MarkdownExample]:
        """Process a code block to extract API references.

        Args:
            block: MarkdownCodeBlock to process

        Returns:
            MarkdownExample if API elements are found, None otherwise
        """
        # Use AST to analyze the code block
        try:
            # Extract imports using AST
            imports = self._extract_imports_from_code(block.code)

            # Find all API elements mentioned in the code
            api_paths = self._find_api_references(block.code, imports)

            if not api_paths:
                return None

            # Build the formatted example with header as comment
            formatted_code = self._format_example(block)

            return MarkdownExample(
                source_file=block.source_file,
                api_paths=api_paths,
                code=formatted_code,
                header=block.preceding_header,
                description=block.preceding_text,
                example_type="readme",
            )
        except Exception as e:
            print(f"Error analyzing code block in {block.source_file}: {e}")
            return None

    def _extract_imports_from_code(self, code: str) -> Dict[str, str]:
        """Extract import statements from code using AST.

        Args:
            code: Source code to analyze

        Returns:
            Dictionary mapping imported names to module paths
        """
        imports = {}

        try:
            # Suppress SyntaxWarnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imports[name] = alias.name

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        # Build full path
                        if module:
                            imports[name] = f"{module}.{alias.name}"
                        else:
                            imports[name] = alias.name

        except SyntaxError:
            # If AST parsing fails, return empty imports
            pass

        return imports

    def _find_api_references(self, code: str, imports: Dict[str, str]) -> Set[str]:
        """Find all API element references in the code.

        Args:
            code: Source code to analyze
            imports: Import mapping from _extract_imports_from_code

        Returns:
            Set of fully qualified API paths found in the code
        """
        api_paths = set()

        # Check imported names
        for imported_name, module_path in imports.items():
            # Try to resolve the module path to a known API element
            resolved = self._resolve_api_path(module_path)
            if resolved:
                api_paths.add(resolved)

            # Also try just the imported name
            resolved = self._resolve_api_path(imported_name)
            if resolved:
                api_paths.add(resolved)

        # Regex-based approach: find potential API references in the code
        # Look for patterns like: ClassName(), module.ClassName(), etc.
        for api_element in self.api_elements:
            parts = api_element.split(".")
            last_part = parts[-1]

            # Check if the last part (class/function name) appears in code
            # Match word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(last_part) + r"\b"
            if re.search(pattern, code):
                # Verify it's actually being used (called or instantiated)
                usage_pattern = r"\b" + re.escape(last_part) + r"\s*\("
                if re.search(usage_pattern, code):
                    api_paths.add(api_element)

        return api_paths

    def _format_example(self, block: MarkdownCodeBlock) -> str:
        """Format a code block as an example with header/description as comments.

        Args:
            block: MarkdownCodeBlock to format

        Returns:
            Formatted code string
        """
        lines = []

        # Add header as comment if present
        if block.preceding_header:
            lines.append(f"# {block.preceding_header}")
            lines.append("")

        # Add description as comment if present
        if block.preceding_text:
            # Wrap long text into multiple comment lines
            text = block.preceding_text
            max_line_length = 80

            words = text.split()
            current_line = "#"

            for word in words:
                if len(current_line) + len(word) + 1 > max_line_length:
                    lines.append(current_line)
                    current_line = f"# {word}"
                else:
                    if current_line == "#":
                        current_line = f"# {word}"
                    else:
                        current_line += f" {word}"

            if current_line != "#":
                lines.append(current_line)

            lines.append("")

        # Add the code
        lines.append(block.code)

        return "\n".join(lines)


def extract_markdown_examples_from_repository(
    repo_path: str, api_elements: Set[str], markdown_paths: Optional[List[str]] = None
) -> List[MarkdownExample]:
    """Convenience function to extract examples from Markdown files in a repository.

    Args:
        repo_path: Path to the repository
        api_elements: Set of known API elements to track
        markdown_paths: Optional list of specific markdown file paths.
                       If None, will search for all .md files in repo.

    Returns:
        List of MarkdownExample objects
    """
    extractor = MarkdownExampleExtractor(repo_path, api_elements)

    # Find markdown files if not provided
    if markdown_paths is None:
        repo_path_obj = Path(repo_path)
        markdown_paths = [str(p) for p in repo_path_obj.rglob("*.md")]

    return extractor.extract_from_markdowns(markdown_paths)
