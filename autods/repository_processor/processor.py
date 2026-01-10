"""Python Repository Processor for generating LLM-ready API documentation."""

import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from autods.constants import REPO_STORAGE
from autods.repository_processor.example_extractor import (
    extract_examples_from_repository,
)
from autods.repository_processor.markdown_example_extractor import (
    extract_markdown_examples_from_repository,
)
from autods.repository_processor.notebook_example_extractor import (
    extract_notebook_examples_from_repository,
)
from autods.repository_processor.utils import (
    RankingStrategy,
    _extract_important_api,
    _extract_test_example_paths,
)
from autods.utils.repo_treesitter import RepoTreeSitter


def clone_repository(repo_url: str, repo_path: str) -> None:
    # Determine if repo_name is a URL or local directory name
    is_url = repo_url.startswith(("http://", "https://", "git@", "git://"))

    # Construct repository path
    if not is_url:
        raise RuntimeError(
            f"Failed to clone repository from {repo_url}. Please provide a valid git URL."
        )

    # Ensure REPO_STORAGE directory exists
    Path(REPO_STORAGE).mkdir(parents=True, exist_ok=True)

    # Clone the repository
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to clone repository from {repo_url}: {e.stderr}. "
            f"Please provide a valid git URL or ensure the repository exists."
        )


async def process_repository(
    repository_path: str,
    output_file: str = "llm-api.xml",
    ranking_strategy: RankingStrategy = "multi_factor_scoring",
    top_n: int = 15,
) -> str:
    """
    Identifies the most important files in a repository using statistical import analysis.
    It then processes these important files using the repository processor.

    Args:
        repository_path: Path to the repository to analyze
        output_file: Name of the output file (default: "llm-api.xml")
        ranking_strategy: Strategy to rank files ("least_internal_imports" or "multi_factor_scoring")
        top_n: Number of top important files to include (default: 10)

    Returns:
        A formatted string with the analysis results and processed files information
    """
    try:
        # Validate repository path
        repo_path = Path(repository_path)
        if not repo_path.exists():
            raise RuntimeError(f"Repository path {repository_path} does not exist")

        important_files_with_scores = _extract_important_api(
            repo_path, strategy=ranking_strategy, top_n=top_n
        )
        if not important_files_with_scores:
            raise RuntimeError("No important files found in the repository")

        # Process repository with path filters
        processor = PythonRepositoryProcessor(str(repo_path))
        classes, functions = processor.process_repository_data()

        # Extract usage examples from repository
        api_usage_groups = extract_examples_from_repository(
            str(repo_path), processor.analysis_results
        )

        # Extract notebook examples and merge into api_usage_groups
        notebook_examples = processor._extract_notebook_examples(classes, functions)
        processor._merge_notebook_examples(api_usage_groups, notebook_examples)

        # Extract markdown examples and merge into api_usage_groups
        markdown_examples = processor._extract_markdown_examples(classes, functions)
        processor._merge_markdown_examples(api_usage_groups, markdown_examples)

        # Merge examples into classes and functions
        processor._merge_examples_into_data(classes, functions, api_usage_groups)

        # Save the results
        output_file = processor.save_repository_data(
            classes, functions, important_files_with_scores, output_file
        )

        # Format results
        result = f"Important Files Analysis Complete\n"
        result += "=" * 40 + "\n\n"
        result += f"Repository: {repository_path}\n"
        result += f"Important files identified: {len(important_files_with_scores)}\n"
        for i, (file_path, score) in enumerate(important_files_with_scores, 1):
            result += f"{i}. {file_path} (score: {int(round(score))})\n"
        result += f"\nProcessed output saved to: {output_file}\n"

        return result

    except Exception as e:
        return f"Error executing repository processor tool: {str(e)}"


@dataclass
class FunctionInfo:
    """Information about a function or method."""

    name: str
    api_path: str
    description: str
    header: str
    output: str
    usage_examples: List[str]


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    api_path: str
    description: str
    initialization: Dict[str, str]
    methods: List[FunctionInfo]
    usage_examples: List[str]


class PythonRepositoryProcessor:
    """Processes Python repositories to generate LLM-ready API documentation."""

    def __init__(self, repo_path: str):
        """Initialize the processor with a repository path.

        Args:
            repo_path: Path to the Python repository to process
        """
        self.repo_path = Path(repo_path)
        self.treesitter = RepoTreeSitter(str(self.repo_path))
        self.analysis_results: Dict[
            str, Any
        ] = {}  # Store analysis results for example extraction

    def _is_under_excluded(self, path, excluded):
        p = Path(path.replace(".", "/"))
        return any(p == exc or exc in p.parents for exc in excluded)

    def process_repository_data(self) -> tuple[List[ClassInfo], List[FunctionInfo]]:
        """Process the repository and return structured data.

        Returns:
            Tuple of (classes, functions) lists containing the processed repository data
        """
        # Analyze the repository structure
        self.analysis_results = self.treesitter.analyze_directory(str(self.repo_path))

        # Process the analysis results into structured data
        classes, functions = self._process_analysis_results(self.analysis_results)

        # Exclude test and example dirs
        paths = _extract_test_example_paths(self.repo_path)
        exclusions = [
            Path(p).relative_to(self.repo_path)
            for p in paths["test"] + paths["example"]
        ]
        classes = [
            p for p in classes if not self._is_under_excluded(p.api_path, exclusions)
        ]
        functions = [
            p for p in functions if not self._is_under_excluded(p.api_path, exclusions)
        ]

        return classes, functions

    def save_repository_data(
        self,
        classes: List[ClassInfo],
        functions: List[FunctionInfo],
        important_files: List[tuple[str, float]],
        output_file: str = "llm-api.xml",
    ) -> str:
        """Save processed repository data to an output file.

        Args:
            classes: List of processed class information
            functions: List of processed function information
            important_files: List of tuples (file_path, score) for important files
            output_file: Name of the output file (default: "llm-api.xml")

        Returns:
            Path to the generated output file
        """
        # Generate XML output
        xml_content = self._generate_xml(classes, functions, important_files)

        # Write to output file
        output_path = self.repo_path / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        return str(output_path)

    def process_repository(self, output_file: str = "llm-api.xml") -> str:
        """Process the repository and generate LLM-ready API documentation.

        Args:
            output_file: Name of the output file (default: "llm-api.xml")

        Returns:
            Path to the generated output file
        """
        classes, functions = self.process_repository_data()
        return self.save_repository_data(classes, functions, [], output_file)

    def _process_analysis_results(
        self, analysis_results: Dict[str, Any]
    ) -> tuple[List[ClassInfo], List[FunctionInfo]]:
        """Process tree-sitter analysis results into structured class and function info.

        Args:
            analysis_results: Results from RepoTreeSitter.analyze_directory()

        Returns:
            Tuple of (classes, functions) lists
        """
        classes = []
        functions = []

        for filename, file_structure in analysis_results.items():
            module_path = self._get_module_path(filename)

            for item in file_structure.get("structure", []):
                if item["type"] == "class":
                    class_info = self._process_class(item, module_path)
                    classes.append(class_info)
                elif item["type"] == "function":
                    function_name = item["details"]["method_name"]
                    # Filter out private functions starting with _
                    if not function_name.startswith("_"):
                        function_info = self._process_function(
                            item["details"], module_path
                        )
                        functions.append(function_info)

        return classes, functions

    def _extract_notebook_examples(
        self, classes: List[ClassInfo], functions: List[FunctionInfo]
    ) -> List[Any]:
        """Extract examples from Jupyter notebooks in the repository.

        Args:
            classes: List of ClassInfo objects to get API paths from
            functions: List of FunctionInfo objects to get API paths from

        Returns:
            List of NotebookExample objects
        """
        # Build set of API elements
        api_elements = set()

        for func_obj in functions:
            api_elements.add(func_obj.api_path)

        for cls_obj in classes:
            api_elements.add(cls_obj.api_path)
            for method in cls_obj.methods:
                api_elements.add(method.api_path)

        # print(api_elements)

        # Find notebook paths
        paths = _extract_test_example_paths(self.repo_path)
        example_paths = paths["example"]

        notebook_paths = []
        for example_path in example_paths:
            example_path_obj = Path(example_path)
            notebook_paths.extend([str(p) for p in example_path_obj.rglob("*.ipynb")])

        # Extract examples if notebooks exist
        if notebook_paths:
            return extract_notebook_examples_from_repository(
                str(self.repo_path), api_elements, notebook_paths
            )

        return []

    def _extract_markdown_examples(
        self, classes: List[ClassInfo], functions: List[FunctionInfo]
    ) -> List[Any]:
        """Extract examples from Markdown files in the repository.

        Args:
            classes: List of ClassInfo objects to get API paths from
            functions: List of FunctionInfo objects to get API paths from

        Returns:
            List of MarkdownExample objects
        """
        # Build set of API elements
        api_elements = set()

        for func_obj in functions:
            api_elements.add(func_obj.api_path)

        for cls_obj in classes:
            api_elements.add(cls_obj.api_path)
            for method in cls_obj.methods:
                api_elements.add(method.api_path)

        # Find all markdown files in the repository
        markdown_paths = [str(p) for p in self.repo_path.rglob("*.md")]

        # Extract examples if markdown files exist
        if markdown_paths:
            return extract_markdown_examples_from_repository(
                str(self.repo_path), api_elements, markdown_paths
            )

        return []

    def _merge_markdown_examples(
        self, api_usage_groups: Dict[str, Any], markdown_examples: List[Any]
    ) -> None:
        """Merge markdown examples into API usage groups.

        Args:
            api_usage_groups: Dictionary mapping API paths to APIUsageGroup objects
            markdown_examples: List of MarkdownExample objects
        """
        from autods.repository_processor.example_extractor import (
            APIUsageGroup,
            UsageExample,
        )

        for markdown_example in markdown_examples:
            # Each markdown example can reference multiple API paths
            for api_path in markdown_example.api_paths:
                # Create APIUsageGroup if it doesn't exist
                if api_path not in api_usage_groups:
                    api_usage_groups[api_path] = APIUsageGroup(
                        api_path=api_path, examples=[]
                    )

                # Create UsageExample
                usage_example = UsageExample(
                    source_file=markdown_example.source_file,
                    function_name=f"readme_{markdown_example.header.lower().replace(' ', '_') if markdown_example.header else 'example'}",
                    source_code=markdown_example.code,
                    start_line=0,
                    used_api_elements=markdown_example.api_paths,
                    example_type="readme",
                    docstring=markdown_example.description,
                )

                # Add header as an attribute for XML generation
                usage_example.header = markdown_example.header

                api_usage_groups[api_path].examples.append(usage_example)
                api_usage_groups[api_path].total_usage_count += 1

    def _merge_notebook_examples(
        self, api_usage_groups: Dict[str, Any], notebook_examples: List[Any]
    ) -> None:
        """Merge notebook examples into API usage groups.

        Args:
            api_usage_groups: Dictionary mapping API paths to APIUsageGroup objects
            notebook_examples: List of NotebookExample objects
        """
        from autods.repository_processor.example_extractor import (
            APIUsageGroup,
            UsageExample,
        )
        from autods.repository_processor.notebook_example_extractor import (
            NotebookExampleExtractor,
        )

        # Create a temporary extractor to use its format method
        temp_extractor = NotebookExampleExtractor(str(self.repo_path), set())

        for notebook_example in notebook_examples:
            api_path = notebook_example.api_path

            # Create APIUsageGroup if it doesn't exist
            if api_path not in api_usage_groups:
                api_usage_groups[api_path] = APIUsageGroup(
                    api_path=api_path, examples=[]
                )

            # Convert NotebookExample to UsageExample format
            # Don't include headers since they'll be in XML attributes
            formatted_source = temp_extractor.format_example(
                notebook_example, include_headers=False
            )

            # Extract variable name from track info
            variable_name = (
                notebook_example.track_info.variable_name
                if notebook_example.track_info
                else None
            )

            usage_example = UsageExample(
                source_file=notebook_example.source_file,
                function_name=f"notebook_{variable_name if variable_name else 'example'}",
                source_code=formatted_source,
                start_line=0,
                used_api_elements={api_path},
                example_type="notebook",
                docstring=None,
            )

            # Add variable_name as an attribute for XML generation
            usage_example.variable_name = variable_name

            api_usage_groups[api_path].examples.append(usage_example)
            api_usage_groups[api_path].total_usage_count += 1

    def _merge_examples_into_data(
        self,
        classes: List[ClassInfo],
        functions: List[FunctionInfo],
        api_usage_groups: Dict[str, Any],
    ) -> None:
        """Merge extracted usage examples into class and function data.

        Args:
            classes: List of ClassInfo objects to update
            functions: List of FunctionInfo objects to update
            api_usage_groups: Dictionary mapping API paths to APIUsageGroup objects
        """
        # Merge examples into functions
        for function in functions:
            if function.api_path in api_usage_groups:
                usage_group = api_usage_groups[function.api_path]
                # Convert UsageExample objects to formatted strings
                function.usage_examples = [
                    self._format_usage_example(example)
                    for example in usage_group.examples
                ]

        # Merge examples into classes and their methods
        for class_info in classes:
            # Add examples for the class itself
            if class_info.api_path in api_usage_groups:
                usage_group = api_usage_groups[class_info.api_path]
                class_info.usage_examples = [
                    self._format_usage_example(example)
                    for example in usage_group.examples
                ]

            # Add examples for each method
            for method in class_info.methods:
                if method.api_path in api_usage_groups:
                    usage_group = api_usage_groups[method.api_path]
                    method.usage_examples = [
                        self._format_usage_example(example)
                        for example in usage_group.examples
                    ]

    def _format_usage_example(self, example: Any) -> str:
        """Format a UsageExample object into a JSON string with metadata.

        Args:
            example: UsageExample object

        Returns:
            JSON string with metadata (from, type, line, variable, header) and source_code
        """
        example_dict = {
            "from": example.source_file,
            "type": example.example_type,
            "line": example.start_line,
            "variable": getattr(example, "variable_name", None),
            "header": getattr(example, "header", None),
            "source_code": example.source_code,
        }
        return json.dumps(example_dict)

    def _get_module_path(self, filename: str) -> str:
        """Generate module path from filename.

        Args:
            filename: Full path to the Python file

        Returns:
            Module path in dot notation (e.g., 'module.submodule')
        """
        # Convert absolute path to relative path from repo root
        rel_path = Path(filename).relative_to(self.repo_path)

        # Remove .py extension and convert path separators to dots
        module_parts = list(rel_path.parts)[:-1] + [rel_path.stem]

        # Filter out __init__ and empty parts
        module_parts = [part for part in module_parts if part and part != "__init__"]

        return ".".join(module_parts)

    def _process_class(self, class_item: Dict[str, Any], module_path: str) -> ClassInfo:
        """Process a class item from tree-sitter analysis.

        Args:
            class_item: Class information from tree-sitter
            module_path: Module path for this class

        Returns:
            ClassInfo object
        """
        class_name = class_item["name"]
        api_path = f"{module_path}.{class_name}" if module_path else class_name
        description = self._clean_docstring(class_item.get("docstring", ""))

        # Process initialization (constructor)
        init_method = None
        for method in class_item.get("methods", []):
            if method["method_name"] == "__init__":
                init_method = method
                break

        initialization = {
            "parameters": self._format_parameters(
                init_method["arguments"] if init_method else []
            ),
            "description": self._clean_docstring(
                init_method["docstring"]
                if init_method and init_method["docstring"]
                else ""
            ),
        }

        # Process methods (excluding __init__ and private methods starting with _)
        methods = []
        for method in class_item.get("methods", []):
            method_name = method["method_name"]
            if method_name != "__init__" and not method_name.startswith("_"):
                method_info = self._process_method(method, api_path)
                methods.append(method_info)

        return ClassInfo(
            name=class_name,
            api_path=api_path,
            description=description,
            initialization=initialization,
            methods=methods,
            usage_examples=[],  # Empty for now as per requirements
        )

    def _process_function(
        self, function_item: Dict[str, Any], module_path: str
    ) -> FunctionInfo:
        """Process a function item from tree-sitter analysis.

        Args:
            function_item: Function information from tree-sitter
            module_path: Module path for this function

        Returns:
            FunctionInfo object
        """
        function_name = function_item["method_name"]
        api_path = f"{module_path}.{function_name}" if module_path else function_name
        description = self._clean_docstring(function_item.get("docstring", ""))

        # Create function header
        params = self._format_parameters(function_item.get("arguments", []))
        return_type = function_item.get("return_type", "")
        header = f"def {function_name}({params})"
        if return_type:
            header += f" -> {return_type}"

        return FunctionInfo(
            name=function_name,
            api_path=api_path,
            description=description,
            header=header,
            output="",  # Empty for now - could be extracted from docstring
            usage_examples=[],  # Empty for now as per requirements
        )

    def _process_method(
        self, method_item: Dict[str, Any], class_api_path: str
    ) -> FunctionInfo:
        """Process a method item from tree-sitter analysis.

        Args:
            method_item: Method information from tree-sitter
            class_api_path: API path of the containing class

        Returns:
            FunctionInfo object
        """
        method_name = method_item["method_name"]
        api_path = f"{class_api_path}.{method_name}"
        description = self._clean_docstring(method_item.get("docstring", ""))

        # Create method header
        params = self._format_parameters(method_item.get("arguments", []))
        return_type = method_item.get("return_type", "")
        header = f"def {method_name}({params})"
        if return_type:
            header += f" -> {return_type}"

        return FunctionInfo(
            name=method_name,
            api_path=api_path,
            description=description,
            header=header,
            output="",  # Empty for now - could be extracted from docstring
            usage_examples=[],  # Empty for now as per requirements
        )

    def _format_parameters(self, arguments: List[str]) -> str:
        """Format function/method parameters.

        Args:
            arguments: List of argument names

        Returns:
            Formatted parameter string
        """
        return ", ".join(arguments)

    def _clean_docstring(self, docstring: Optional[str]) -> str:
        """Clean and format docstring.

        Args:
            docstring: Raw docstring from tree-sitter

        Returns:
            Cleaned docstring
        """
        if not docstring:
            return ""

        # Remove triple quotes and clean up formatting
        cleaned = docstring.strip()
        if cleaned.startswith('"""') or cleaned.startswith("'''"):
            cleaned = cleaned[3:]
        if cleaned.endswith('"""') or cleaned.endswith("'''"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    def _generate_xml(
        self,
        classes: List[ClassInfo],
        functions: List[FunctionInfo],
        important_files: List[tuple[str, float]],
    ) -> str:
        """Generate XML output from processed class and function information.

        Args:
            classes: List of processed class information
            functions: List of processed function information
            important_files: List of tuples (file_path, score) for important files

        Returns:
            XML string
        """
        root = ET.Element("repository")

        # Add important files with scores
        important_files_elem = ET.SubElement(root, "important_files")
        for file_path, score in important_files:
            file_elem = ET.SubElement(important_files_elem, "file")
            file_elem.set("score", str(int(round(score))))
            file_elem.text = file_path

        # Add classes
        for class_info in classes:
            class_elem = ET.SubElement(root, "class")

            ET.SubElement(class_elem, "name").text = class_info.name
            ET.SubElement(class_elem, "api_path").text = class_info.api_path
            ET.SubElement(class_elem, "description").text = class_info.description

            # Initialization
            init_elem = ET.SubElement(class_elem, "initialization")
            ET.SubElement(init_elem, "parameters").text = class_info.initialization[
                "parameters"
            ]
            ET.SubElement(init_elem, "description").text = class_info.initialization[
                "description"
            ]

            # Methods
            methods_elem = ET.SubElement(class_elem, "methods")
            for method in class_info.methods:
                method_elem = ET.SubElement(methods_elem, "method")
                ET.SubElement(method_elem, "name").text = method.name
                ET.SubElement(method_elem, "api_path").text = method.api_path
                ET.SubElement(method_elem, "description").text = method.description
                ET.SubElement(method_elem, "header").text = method.header
                ET.SubElement(method_elem, "output").text = method.output

                # Usage examples
                usage_examples_elem = ET.SubElement(method_elem, "usage_examples")
                for example in method.usage_examples:
                    example_elem = ET.SubElement(usage_examples_elem, "example")
                    # Try to parse as JSON first, fall back to dict or plain string
                    example_dict: Dict[str, Any] | None = None
                    if isinstance(example, dict):
                        example_dict = example
                    elif isinstance(example, str):
                        try:
                            example_dict = json.loads(example)
                        except (json.JSONDecodeError, TypeError):
                            # Legacy format (plain string)
                            example_elem.text = example
                            continue

                    if example_dict is not None:
                        # Structured format
                        if example_dict.get("from"):
                            ET.SubElement(example_elem, "from").text = str(
                                example_dict["from"]
                            )
                        if example_dict.get("type"):
                            ET.SubElement(example_elem, "type").text = str(
                                example_dict["type"]
                            )
                        if example_dict.get("line") is not None:
                            ET.SubElement(example_elem, "line").text = str(
                                example_dict["line"]
                            )
                        if example_dict.get("variable"):
                            ET.SubElement(example_elem, "variable").text = str(
                                example_dict["variable"]
                            )
                        if example_dict.get("header"):
                            ET.SubElement(example_elem, "header").text = str(
                                example_dict["header"]
                            )
                        ET.SubElement(
                            example_elem, "source_code"
                        ).text = example_dict.get("source_code", "")

            # Class usage examples
            class_usage_examples_elem = ET.SubElement(class_elem, "usage_examples")
            for example in class_info.usage_examples:
                example_elem = ET.SubElement(class_usage_examples_elem, "example")
                # Try to parse as JSON first, fall back to dict or plain string
                class_example_dict: Dict[str, Any] | None = None
                if isinstance(example, dict):
                    class_example_dict = example
                elif isinstance(example, str):
                    try:
                        class_example_dict = json.loads(example)
                    except (json.JSONDecodeError, TypeError):
                        # Legacy format (plain string)
                        example_elem.text = example
                        continue

                if class_example_dict is not None:
                    # Structured format
                    if class_example_dict.get("from"):
                        ET.SubElement(example_elem, "from").text = str(
                            class_example_dict["from"]
                        )
                    if class_example_dict.get("type"):
                        ET.SubElement(example_elem, "type").text = str(
                            class_example_dict["type"]
                        )
                    if class_example_dict.get("line") is not None:
                        ET.SubElement(example_elem, "line").text = str(
                            class_example_dict["line"]
                        )
                    if class_example_dict.get("variable"):
                        ET.SubElement(example_elem, "variable").text = str(
                            class_example_dict["variable"]
                        )
                    if class_example_dict.get("header"):
                        ET.SubElement(example_elem, "header").text = str(
                            class_example_dict["header"]
                        )
                    ET.SubElement(
                        example_elem, "source_code"
                    ).text = class_example_dict.get("source_code", "")

        # Add functions
        for function_info in functions:
            function_elem = ET.SubElement(root, "function")

            ET.SubElement(function_elem, "name").text = function_info.name
            ET.SubElement(function_elem, "api_path").text = function_info.api_path
            ET.SubElement(function_elem, "description").text = function_info.description
            ET.SubElement(function_elem, "header").text = function_info.header
            ET.SubElement(function_elem, "output").text = function_info.output

            # Usage examples
            usage_examples_elem = ET.SubElement(function_elem, "usage_examples")
            for example in function_info.usage_examples:
                example_elem = ET.SubElement(usage_examples_elem, "example")
                # Try to parse as JSON first, fall back to dict or plain string
                func_example_dict: Dict[str, Any] | None = None
                if isinstance(example, dict):
                    func_example_dict = example
                elif isinstance(example, str):
                    try:
                        func_example_dict = json.loads(example)
                    except (json.JSONDecodeError, TypeError):
                        # Legacy format (plain string)
                        example_elem.text = example
                        continue

                if func_example_dict is not None:
                    # Structured format
                    if func_example_dict.get("from"):
                        ET.SubElement(example_elem, "from").text = str(
                            func_example_dict["from"]
                        )
                    if func_example_dict.get("type"):
                        ET.SubElement(example_elem, "type").text = str(
                            func_example_dict["type"]
                        )
                    if func_example_dict.get("line") is not None:
                        ET.SubElement(example_elem, "line").text = str(
                            func_example_dict["line"]
                        )
                    if func_example_dict.get("variable"):
                        ET.SubElement(example_elem, "variable").text = str(
                            func_example_dict["variable"]
                        )
                    if func_example_dict.get("header"):
                        ET.SubElement(example_elem, "header").text = str(
                            func_example_dict["header"]
                        )
                    ET.SubElement(
                        example_elem, "source_code"
                    ).text = func_example_dict.get("source_code", "")

        # Convert to string with formatting
        rough_string = ET.tostring(root, encoding="unicode")
        return self._prettify_xml(rough_string)

    def _sanitize_xml_text(self, text: str) -> str:
        """Remove invalid XML characters from text.

        Args:
            text: Text that may contain invalid XML characters

        Returns:
            Sanitized text safe for XML
        """
        import re

        if not text:
            return text

        # Remove control characters except tab (0x09), newline (0x0A), carriage return (0x0D)
        # Valid XML chars: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD]
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    def _prettify_xml(self, xml_string: str) -> str:
        """Prettify XML string with proper indentation.

        Args:
            xml_string: Raw XML string

        Returns:
            Formatted XML string
        """
        from xml.dom import minidom

        # Sanitize the XML string before parsing
        xml_string = self._sanitize_xml_text(xml_string)

        try:
            dom = minidom.parseString(xml_string)
            return dom.toprettyxml(indent="  ", encoding=None)
        except Exception as e:
            # If prettification fails, return the raw XML with a warning
            print(f"Warning: XML prettification failed: {e}")
            print("Returning unprettified XML")
            return xml_string
