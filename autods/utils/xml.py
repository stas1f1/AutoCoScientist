"""Programmable API tool for extracting LLM-API documentation sections."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from autods.tools.base import ToolError

MAX_USAGE_EXAMPLES = 8
MAX_EXAMPLE_LINES = 100


def extract_entity_section(
    llm_api_path: Path, entity_name: str, output_format: str = "markdown"
) -> Optional[str]:
    """Extract a specific entity (class or function) section from llm-api.xml.

    Args:
        llm_api_path: Path to the llm-api.xml file
        entity_name: Name of the entity to extract
        output_format: Output format - 'markdown' or 'xml'

    Returns:
        Markdown-formatted string or raw XML with the entity's documentation section, or None if not found
    """
    try:
        tree = ET.parse(llm_api_path)
        root = tree.getroot()

        # Search for the entity in classes
        for class_elem in root.findall("class"):
            name_elem = class_elem.find("name")
            if name_elem is not None and name_elem.text == entity_name:
                if output_format == "xml":
                    return ET.tostring(class_elem, encoding="unicode", method="xml")
                return format_class_section(class_elem)

            # Also search in class methods
            methods_elem = class_elem.find("methods")
            if methods_elem is not None:
                for method_elem in methods_elem.findall("method"):
                    method_name_elem = method_elem.find("name")
                    if (
                        method_name_elem is not None
                        and method_name_elem.text == entity_name
                    ):
                        if output_format == "xml":
                            return ET.tostring(
                                method_elem, encoding="unicode", method="xml"
                            )
                        class_name: str = (
                            name_elem.text
                            if name_elem is not None and name_elem.text is not None
                            else "Unknown"
                        )
                        return format_method_section(method_elem, class_name)

        # Search for the entity in functions
        for function_elem in root.findall("function"):
            name_elem = function_elem.find("name")
            if name_elem is not None and name_elem.text == entity_name:
                if output_format == "xml":
                    return ET.tostring(function_elem, encoding="unicode", method="xml")
                return format_function_section(function_elem)

        return None

    except Exception as e:
        raise ToolError(f"Error parsing llm-api.xml: {str(e)}")


def format_class_section(class_elem: ET.Element) -> str:
    """Format a class element into a markdown string.

    Args:
        class_elem: XML element representing a class

    Returns:
        Markdown-formatted string with class documentation
    """
    result = []
    result.append("# Class Documentation")
    result.append("")

    # Basic info
    name = get_text_from_element(class_elem, "name", "Unknown")
    api_path = get_text_from_element(class_elem, "api_path", "Unknown")
    description = get_text_from_element(
        class_elem, "description", "No description available"
    )

    result.append(f"**Name:** `{name}`")
    result.append(f"**API Path:** `{api_path}`")
    result.append("")
    result.append("## Description")
    result.append(description)
    result.append("")

    # Initialization
    init_elem = class_elem.find("initialization")
    if init_elem is not None:
        params = get_text_from_element(init_elem, "parameters", "")
        init_desc = get_text_from_element(init_elem, "description", "")

        result.append("## Initialization")
        result.append("```python")
        result.append(f"__init__({params})")
        result.append("```")
        if init_desc:
            result.append(init_desc)
        result.append("")

    # Methods
    methods_elem = class_elem.find("methods")
    if methods_elem is not None:
        methods = methods_elem.findall("method")
        if methods:
            result.append(f"## Methods ({len(methods)})")
            result.append("")
            for method in methods:
                method_name = get_text_from_element(method, "name", "unknown")
                method_header = get_text_from_element(method, "header", "")
                method_desc = get_text_from_element(
                    method, "description", "No description"
                )

                result.append(f"### {method_name}")
                result.append("")
                result.append("**Signature:**")
                result.append("```python")
                result.append(method_header)
                result.append("```")
                result.append("")
                result.append("**Description:**")
                result.append(method_desc)
                result.append("")

                # Add usage examples if available
                usage_examples = format_usage_examples(method)
                if usage_examples:
                    result.append("**Usage Examples:**")
                    result.append("")
                    for example in usage_examples:
                        # Indent example content properly
                        for line in example.split("\n"):
                            result.append(line)
                    result.append("")

    # Class-level usage examples
    class_examples = format_usage_examples(class_elem)
    if class_examples:
        result.append("## Class Usage Examples")
        result.append("")
        for example in class_examples:
            for line in example.split("\n"):
                result.append(line)
        result.append("")

    return "\n".join(result)


def format_function_section(function_elem: ET.Element) -> str:
    """Format a function element into a markdown string.

    Args:
        function_elem: XML element representing a function

    Returns:
        Markdown-formatted string with function documentation
    """
    result = []
    result.append("# Function Documentation")
    result.append("")

    # Basic info
    name = get_text_from_element(function_elem, "name", "Unknown")
    api_path = get_text_from_element(function_elem, "api_path", "Unknown")
    description = get_text_from_element(
        function_elem, "description", "No description available"
    )
    header = get_text_from_element(function_elem, "header", "")
    output = get_text_from_element(function_elem, "output", "")

    result.append(f"**Name:** `{name}`")
    result.append(f"**API Path:** `{api_path}`")
    result.append("")
    result.append("## Signature")
    result.append("```python")
    result.append(header)
    result.append("```")
    result.append("")
    result.append("## Description")
    result.append(description)
    result.append("")

    if output:
        result.append("## Output")
        result.append(output)
        result.append("")

    # Usage examples
    usage_examples = format_usage_examples(function_elem)
    if usage_examples:
        result.append("## Usage Examples")
        result.append("")
        for example in usage_examples:
            for line in example.split("\n"):
                result.append(line)
        result.append("")

    return "\n".join(result)


def format_method_section(method_elem: ET.Element, class_name: str) -> str:
    """Format a method element into a markdown string.

    Args:
        method_elem: XML element representing a method
        class_name: Name of the containing class

    Returns:
        Markdown-formatted string with method documentation
    """
    result = []
    result.append("# Method Documentation")
    result.append("")

    # Basic info
    name = get_text_from_element(method_elem, "name", "Unknown")
    api_path = get_text_from_element(method_elem, "api_path", "Unknown")
    description = get_text_from_element(
        method_elem, "description", "No description available"
    )
    header = get_text_from_element(method_elem, "header", "")
    output = get_text_from_element(method_elem, "output", "")

    result.append(f"**Class:** `{class_name}`")
    result.append(f"**Method Name:** `{name}`")
    result.append(f"**API Path:** `{api_path}`")
    result.append("")
    result.append("## Signature")
    result.append("```python")
    result.append(header)
    result.append("```")
    result.append("")
    result.append("## Description")
    result.append(description)
    result.append("")

    if output:
        result.append("## Output")
        result.append(output)
        result.append("")

    # Usage examples
    usage_examples = format_usage_examples(method_elem)
    if usage_examples:
        result.append("## Usage Examples")
        result.append("")
        for example in usage_examples:
            for line in example.split("\n"):
                result.append(line)
        result.append("")

    return "\n".join(result)


def format_usage_examples(elem: ET.Element) -> list[str]:
    """Extract and format usage examples from an element as markdown.

    Examples are sorted by type importance (readme > notebook > example > test)
    and capped at MAX_USAGE_EXAMPLES. Individual examples are cropped to
    MAX_EXAMPLE_LINES.

    Args:
        elem: XML element containing usage_examples

    Returns:
        List of markdown-formatted usage example strings
    """
    # Define type importance ranking (lower number = higher priority)
    type_priority = {
        "readme": 0,
        "markdown": 0,  # Treat markdown same as readme
        "notebook": 1,
        "example": 2,
        "test": 3,
    }

    examples_with_priority = []
    usage_examples_elem = elem.find("usage_examples")

    if usage_examples_elem is not None:
        for example_elem in usage_examples_elem.findall("example"):
            # Check if it's structured format
            from_elem = example_elem.find("from")
            type_elem = example_elem.find("type")
            source_code_elem = example_elem.find("source_code")

            # Get example type for sorting
            example_type = (
                type_elem.text.lower()
                if type_elem is not None and type_elem.text
                else "test"
            )
            priority = type_priority.get(
                example_type, 999
            )  # Unknown types get lowest priority

            if from_elem is not None or source_code_elem is not None:
                # Structured format
                example_parts = []
                if from_elem is not None and from_elem.text:
                    example_parts.append(f"**From:** `{from_elem.text}`")
                if type_elem is not None and type_elem.text:
                    example_parts.append(f"**Type:** `{type_elem.text}`")

                line_elem = example_elem.find("line")
                if line_elem is not None and line_elem.text:
                    example_parts.append(f"**Line:** `{line_elem.text}`")

                variable_elem = example_elem.find("variable")
                if variable_elem is not None and variable_elem.text:
                    example_parts.append(f"**Variable:** `{variable_elem.text}`")

                header_elem = example_elem.find("header")
                if header_elem is not None and header_elem.text:
                    example_parts.append(f"**Header:** `{header_elem.text}`")

                if source_code_elem is not None and source_code_elem.text:
                    example_parts.append("")
                    example_parts.append("**Source Code:**")
                    example_parts.append("```python")

                    # Crop source code to MAX_EXAMPLE_LINES
                    source_lines = source_code_elem.text.strip().split("\n")
                    if len(source_lines) > MAX_EXAMPLE_LINES:
                        cropped_lines = source_lines[:MAX_EXAMPLE_LINES]
                        cropped_lines.append(
                            f"... (cropped to {MAX_EXAMPLE_LINES} lines)"
                        )
                        example_parts.append("\n".join(cropped_lines))
                    else:
                        example_parts.append(source_code_elem.text.strip())

                    example_parts.append("```")

                examples_with_priority.append((priority, "\n".join(example_parts)))
            else:
                # Legacy format (plain text) - wrap in code block if it looks like code
                if example_elem.text:
                    text = example_elem.text.strip()

                    # Crop text to MAX_EXAMPLE_LINES
                    text_lines = text.split("\n")
                    if len(text_lines) > MAX_EXAMPLE_LINES:
                        text_lines = text_lines[:MAX_EXAMPLE_LINES]
                        text_lines.append(f"... (cropped to {MAX_EXAMPLE_LINES} lines)")
                        text = "\n".join(text_lines)

                    # Simple heuristic: if it contains parentheses or equals, treat as code
                    if "(" in text or "=" in text:
                        examples_with_priority.append(
                            (priority, f"```python\n{text}\n```")
                        )
                    else:
                        examples_with_priority.append((priority, text))

    # Sort by priority (ascending) and cap at MAX_USAGE_EXAMPLES
    examples_with_priority.sort(key=lambda x: x[0])
    examples = [example for _, example in examples_with_priority[:MAX_USAGE_EXAMPLES]]

    return examples


def get_text_from_element(elem: ET.Element, tag: str, default: str = "") -> str:
    """Safely extract text from an XML element.

    Args:
        elem: Parent XML element
        tag: Tag name to find
        default: Default value if element not found or has no text

    Returns:
        Text content or default value
    """
    child = elem.find(tag)
    if child is not None and child.text:
        return child.text
    return default


def extract_all_entities(llm_api_path: Path) -> list[tuple[str, str, str, dict]]:
    """Extract all outermost blocks (classes and functions) from llm-api.xml with chunking.

    Args:
        llm_api_path: Path to the llm-api.xml file

    Returns:
        List of tuples (entity_name, chunk_type, content_text, metadata) where:
        - entity_name: Name of the parent class or function
        - chunk_type: Type of chunk ('class_overview', 'method', 'function')
        - content_text: Text representation of the chunk
        - metadata: Additional metadata (method_name, etc.)
    """
    try:
        tree = ET.parse(llm_api_path)
        root = tree.getroot()
        chunks: list[tuple[str, str, str, dict[str, Any]]] = []

        # Extract all classes with chunking
        for class_elem in root.findall("class"):
            name = get_text_from_element(class_elem, "name", "Unknown")
            api_path = get_text_from_element(class_elem, "api_path", "")
            description = get_text_from_element(class_elem, "description", "")

            # Chunk 1: Class overview (name, description, initialization)
            overview_parts = [f"Class: {name}"]
            if api_path:
                overview_parts.append(f"API Path: {api_path}")
            if description:
                overview_parts.append(f"Description: {description}")

            # Add initialization info to overview
            init_elem = class_elem.find("initialization")
            if init_elem is not None:
                params = get_text_from_element(init_elem, "parameters", "")
                init_desc = get_text_from_element(init_elem, "description", "")
                if params:
                    overview_parts.append(f"Initialization: __init__({params})")
                if init_desc:
                    overview_parts.append(f"Init Description: {init_desc}")

            chunks.append((name, "class_overview", "\n".join(overview_parts), {}))

            # Chunk 2+: Each method as a separate chunk
            methods_elem = class_elem.find("methods")
            if methods_elem is not None:
                methods = methods_elem.findall("method")
                for method in methods:
                    method_name = get_text_from_element(method, "name", "unknown")
                    method_desc = get_text_from_element(method, "description", "")
                    method_header = get_text_from_element(method, "header", "")
                    method_output = get_text_from_element(method, "output", "")

                    # Build method chunk
                    method_parts = [
                        f"Class: {name}",
                        f"Method: {method_name}",
                    ]
                    if method_header:
                        method_parts.append(f"Signature: {method_header}")
                    if method_desc:
                        method_parts.append(f"Description: {method_desc}")
                    if method_output:
                        method_parts.append(f"Output: {method_output}")

                    chunks.append(
                        (
                            name,
                            "method",
                            "\n".join(method_parts),
                            {"method_name": method_name},
                        )
                    )

        # Extract all functions (no chunking needed, they're already small)
        for function_elem in root.findall("function"):
            name = get_text_from_element(function_elem, "name", "Unknown")
            api_path = get_text_from_element(function_elem, "api_path", "")
            description = get_text_from_element(function_elem, "description", "")
            header = get_text_from_element(function_elem, "header", "")
            output = get_text_from_element(function_elem, "output", "")

            # Build a text representation of the function
            text_parts = [f"Function: {name}"]
            if api_path:
                text_parts.append(f"API Path: {api_path}")
            if header:
                text_parts.append(f"Signature: {header}")
            if description:
                text_parts.append(f"Description: {description}")
            if output:
                text_parts.append(f"Output: {output}")

            chunks.append((name, "function", "\n".join(text_parts), {}))

        return chunks

    except Exception as e:
        raise ToolError(f"Error extracting entities from llm-api.xml: {str(e)}")
