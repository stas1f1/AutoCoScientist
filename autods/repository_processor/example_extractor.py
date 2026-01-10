"""Extract method-level usage examples from test and example codebases.

This module provides functionality to extract self-contained usage examples
from test files and example codebases, tracking in-library function/class usage
and linking them to the API surface.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from autods.repository_processor.utils import _extract_test_example_paths
from autods.utils.repo_treesitter import RepoTreeSitter


@dataclass
class UsageExample:
    """Represents a single usage example extracted from tests or examples."""

    source_file: str
    function_name: str
    source_code: str
    start_line: int
    used_api_elements: Set[str] = field(default_factory=set)
    example_type: str = "test"  # "test" or "example"
    docstring: Optional[str] = None
    header: Optional[str] = None  # Optional header for markdown examples
    variable_name: Optional[str] = None  # Optional variable name for notebook examples


@dataclass
class APIUsageGroup:
    """Groups examples by the API element they demonstrate."""

    api_path: str  # Fully qualified name (e.g., "module.Class.method")
    examples: List[UsageExample] = field(default_factory=list)
    total_usage_count: int = 0


class ExampleExtractor:
    """Extracts method-level usage examples from test and example codebases."""

    def __init__(self, repo_path: str):
        """Initialize the example extractor.

        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)
        self.treesitter = RepoTreeSitter(str(self.repo_path))

        # Maps to track API surface
        self.api_elements: Set[str] = set()  # All public API elements
        self.module_map: Dict[str, str] = {}  # module_name -> file_path

    def extract_examples(
        self,
        project_structure: Dict[str, Any],
        test_paths: List[str],
        example_paths: List[str],
    ) -> Dict[str, APIUsageGroup]:
        """Main method to extract usage examples.

        Args:
            project_structure: AST parsed structure from RepoTreeSitter.analyze_directory
            test_paths: List of test directory paths
            example_paths: List of example directory paths

        Returns:
            Dictionary mapping API paths to their usage groups
        """
        # Step 1: Build API surface map from main codebase
        self._build_api_surface(project_structure, test_paths, example_paths)

        # Step 2: Extract examples from tests
        test_examples = self._extract_from_tests(test_paths)

        # Step 3: Extract examples from example codebases
        example_examples = self._extract_from_examples(example_paths)

        # Step 4: Combine and group by API element
        all_examples = test_examples + example_examples
        grouped = self._group_by_api_element(all_examples)

        return grouped

    def _build_api_surface(
        self,
        project_structure: Dict[str, Any],
        test_paths: List[str],
        example_paths: List[str],
    ) -> None:
        """Build a map of the public API surface.

        Args:
            project_structure: Parsed structure from TreeSitter
            test_paths: Paths to exclude (test directories)
            example_paths: Paths to exclude (example directories)
        """
        exclusion_paths = set(test_paths + example_paths)

        for file_path, file_structure in project_structure.items():
            # Skip test and example files
            if any(str(excl) in file_path for excl in exclusion_paths):
                continue

            # Skip private modules
            if (
                Path(file_path).stem.startswith("_")
                and Path(file_path).stem != "__init__"
            ):
                continue

            module_path = self._get_module_path(file_path)
            self.module_map[module_path] = file_path

            # Extract API elements from this file
            for item in file_structure.get("structure", []):
                if item["type"] == "class":
                    class_name = item["name"]
                    # Skip private classes
                    if class_name.startswith("_"):
                        continue

                    class_api_path = (
                        f"{module_path}.{class_name}" if module_path else class_name
                    )
                    self.api_elements.add(class_api_path)

                    # Add public methods
                    for method in item.get("methods", []):
                        method_name = method["method_name"]
                        if not method_name.startswith("_") or method_name == "__init__":
                            method_api_path = f"{class_api_path}.{method_name}"
                            self.api_elements.add(method_api_path)

                elif item["type"] == "function":
                    function_name = item["details"]["method_name"]
                    # Skip private functions
                    if not function_name.startswith("_"):
                        function_api_path = (
                            f"{module_path}.{function_name}"
                            if module_path
                            else function_name
                        )
                        self.api_elements.add(function_api_path)

    def _extract_from_tests(self, test_paths: List[str]) -> List[UsageExample]:
        """Extract examples from test files.

        Each test function is treated as a self-contained example.

        Args:
            test_paths: List of test directory paths

        Returns:
            List of extracted usage examples
        """
        examples = []

        for test_path in test_paths:
            if not Path(test_path).exists():
                continue

            # Parse all test files
            test_structure = self.treesitter.analyze_directory(test_path)

            for file_path, file_structure in test_structure.items():
                for item in file_structure.get("structure", []):
                    # Extract test functions
                    if item["type"] == "function":
                        function_details = item["details"]
                        function_name = function_details["method_name"]

                        # Only process test functions
                        if not function_name.startswith("test_"):
                            continue

                        # Track used API elements
                        used_apis = self._resolve_used_apis(
                            function_details.get("method_calls", []),
                            file_structure.get("imports", {}),
                        )

                        example = UsageExample(
                            source_file=file_path,
                            function_name=function_name,
                            source_code=function_details.get("source_code", ""),
                            start_line=function_details.get("start_line", 0),
                            used_api_elements=used_apis,
                            example_type="test",
                            docstring=function_details.get("docstring"),
                        )
                        examples.append(example)

                    # Extract test methods from test classes
                    elif item["type"] == "class":
                        for method in item.get("methods", []):
                            method_name = method["method_name"]

                            # Only process test methods
                            if not method_name.startswith("test_"):
                                continue

                            used_apis = self._resolve_used_apis(
                                method.get("method_calls", []),
                                file_structure.get("imports", {}),
                            )

                            example = UsageExample(
                                source_file=file_path,
                                function_name=f"{item['name']}.{method_name}",
                                source_code=method.get("source_code", ""),
                                start_line=method.get("start_line", 0),
                                used_api_elements=used_apis,
                                example_type="test",
                                docstring=method.get("docstring"),
                            )
                            examples.append(example)

        return examples

    def _extract_from_examples(self, example_paths: List[str]) -> List[UsageExample]:
        """Extract examples from example codebases.

        Traces data flow from imports → instantiation → method calls.

        Args:
            example_paths: List of example directory paths

        Returns:
            List of extracted usage examples
        """
        examples = []

        for example_path in example_paths:
            if not Path(example_path).exists():
                continue

            # Parse all example files
            example_structure = self.treesitter.analyze_directory(example_path)

            for file_path, file_structure in example_structure.items():
                # Process functions in example files
                for item in file_structure.get("structure", []):
                    if item["type"] == "function":
                        function_details = item["details"]
                        function_name = function_details["method_name"]

                        # Skip private functions
                        if function_name.startswith("_"):
                            continue

                        # Track used API elements through data flow
                        used_apis = self._trace_data_flow(
                            function_details, file_structure.get("imports", {})
                        )

                        example = UsageExample(
                            source_file=file_path,
                            function_name=function_name,
                            source_code=function_details.get("source_code", ""),
                            start_line=function_details.get("start_line", 0),
                            used_api_elements=used_apis,
                            example_type="example",
                            docstring=function_details.get("docstring"),
                        )
                        examples.append(example)

                    # Process class methods in example files
                    elif item["type"] == "class":
                        for method in item.get("methods", []):
                            method_name = method["method_name"]

                            # Skip private methods
                            if (
                                method_name.startswith("_")
                                and method_name != "__init__"
                            ):
                                continue

                            used_apis = self._trace_data_flow(
                                method, file_structure.get("imports", {})
                            )

                            example = UsageExample(
                                source_file=file_path,
                                function_name=f"{item['name']}.{method_name}",
                                source_code=method.get("source_code", ""),
                                start_line=method.get("start_line", 0),
                                used_api_elements=used_apis,
                                example_type="example",
                                docstring=method.get("docstring"),
                            )
                            examples.append(example)

        return examples

    def _trace_data_flow(
        self, function_details: Dict[str, Any], imports: Dict[str, Any]
    ) -> Set[str]:
        """Trace data flow from imports → instantiation → method calls.

        Args:
            function_details: Function/method details from TreeSitter
            imports: Import mapping from the file

        Returns:
            Set of API paths used in this function
        """
        used_apis = set()

        # Get method calls from the function
        method_calls = function_details.get("method_calls", [])

        for call_info in method_calls:
            # Resolve to fully qualified API path
            api_path = self._resolve_call_to_api_path(call_info, imports)
            if api_path and api_path in self.api_elements:
                used_apis.add(api_path)

        return used_apis

    def _resolve_used_apis(
        self, method_calls: List[Dict[str, Any]], imports: Dict[str, Any]
    ) -> Set[str]:
        """Resolve method calls to API paths.

        Args:
            method_calls: List of method call information
            imports: Import mapping

        Returns:
            Set of resolved API paths
        """
        used_apis = set()

        for call_info in method_calls:
            api_path = self._resolve_call_to_api_path(call_info, imports)
            if api_path and api_path in self.api_elements:
                used_apis.add(api_path)

        return used_apis

    def _resolve_call_to_api_path(
        self, call_info: Dict[str, Any], imports: Dict[str, Any]
    ) -> Optional[str]:
        """Resolve a method call to its fully qualified API path.

        Args:
            call_info: Call information from TreeSitter (contains module, class, function, path)
            imports: Import mapping

        Returns:
            Fully qualified API path or None if not resolvable
        """
        # Extract components from call_info
        module = call_info.get("module", "")
        class_name = call_info.get("class", "")
        function_name = call_info.get("function", "")
        path = call_info.get("path", "")

        # Check if this is from our repository
        if not path or not Path(path).is_relative_to(self.repo_path):
            return None

        # Build the API path
        api_parts = []

        # Get module path from file path
        if path:
            module_path = self._get_module_path(path)
            if module_path:
                api_parts.append(module_path)
        elif module:
            api_parts.append(module)

        if class_name:
            api_parts.append(class_name)

        if function_name:
            # Remove parentheses and chained calls
            clean_function = function_name.split("(")[0].split(".")[0]
            api_parts.append(clean_function)

        if api_parts:
            api_path = ".".join(api_parts)

            # If the constructed path doesn't exist in API surface,
            # try to find it by searching for matching patterns
            # This handles cases where imports are from __init__.py but classes are in submodules
            if api_path not in self.api_elements:
                # Try to find a matching API element that ends with the class.method pattern
                if class_name and function_name:
                    search_suffix = f".{class_name}.{clean_function}"
                    for api_element in self.api_elements:
                        if api_element.endswith(search_suffix):
                            return api_element
                elif class_name:
                    search_suffix = f".{class_name}"
                    for api_element in self.api_elements:
                        if api_element.endswith(search_suffix):
                            return api_element

            return api_path

        return None

    def _group_by_api_element(
        self, examples: List[UsageExample]
    ) -> Dict[str, APIUsageGroup]:
        """Group examples by the API elements they demonstrate.

        Args:
            examples: List of all extracted examples

        Returns:
            Dictionary mapping API paths to usage groups
        """
        grouped: Dict[str, APIUsageGroup] = defaultdict(
            lambda: APIUsageGroup(api_path="", examples=[])
        )

        for example in examples:
            # Add this example to each API element it uses
            for api_path in example.used_api_elements:
                if api_path not in grouped:
                    grouped[api_path] = APIUsageGroup(api_path=api_path, examples=[])

                grouped[api_path].examples.append(example)
                grouped[api_path].total_usage_count += 1

        return dict(grouped)

    def _get_module_path(self, file_path: str) -> str:
        """Generate module path from file path.

        Args:
            file_path: Full path to the Python file

        Returns:
            Module path in dot notation (e.g., 'module.submodule')
        """
        try:
            # Convert absolute path to relative path from repo root
            rel_path = Path(file_path).relative_to(self.repo_path)

            # Remove .py extension and convert path separators to dots
            module_parts = list(rel_path.parts)[:-1] + [rel_path.stem]

            # Filter out __init__ and empty parts
            module_parts = [
                part for part in module_parts if part and part != "__init__"
            ]

            return ".".join(module_parts)
        except (ValueError, AttributeError):
            return ""


def extract_examples_from_repository(
    repo_path: str, project_structure: Optional[Dict[str, Any]] = None
) -> Dict[str, APIUsageGroup]:
    """Convenience function to extract examples from a repository.

    Args:
        repo_path: Path to the repository
        project_structure: Optional pre-parsed project structure.
                          If None, will parse the entire repository.

    Returns:
        Dictionary mapping API paths to their usage groups
    """
    extractor = ExampleExtractor(repo_path)

    # Get test and example paths
    paths = _extract_test_example_paths(Path(repo_path))
    test_paths = paths["test"]
    example_paths = paths["example"]

    # Parse project structure if not provided
    if project_structure is None:
        treesitter = RepoTreeSitter(repo_path)
        project_structure = treesitter.analyze_directory(repo_path)

    # Extract examples
    return extractor.extract_examples(project_structure, test_paths, example_paths)
