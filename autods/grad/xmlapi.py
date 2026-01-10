import xml.etree.ElementTree as ET
from pathlib import Path


def get_text(elem: ET.Element, tag: str, default: str = "") -> str:
    child = elem.find(tag)
    if child is not None and child.text:
        return child.text
    return default


def get_examples(elem: ET.Element) -> list[str]:
    examples_elem = elem.find("usage_examples")
    examples = []
    if examples_elem:
        for example_elem in examples_elem.findall("example"):
            from_ = get_text(example_elem, "from", "")
            type_ = get_text(example_elem, "type", "")
            source_code_ = get_text(example_elem, "source_code", "")

            if source_code_.strip():
                examples.append(
                    f'<example from="{from_}" type="{type_}">\n{source_code_}\n</example>'
                )
    return examples


def extract_entities(
    xml_api_path: Path,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Extract top-level entities: classes, methods, functions, examples

    Args:
        xml_api_path: Path to the api.xml file
    """
    try:
        tree = ET.parse(xml_api_path)
        root = tree.getroot()

        # Top-level Entities
        classes: list[str] = []
        methods: list[str] = []
        functions: list[str] = []
        examples: list[str] = []

        # Extract all classes
        for class_elem in root.findall("class"):
            name = get_text(class_elem, "name", "Unknown")
            api_path = get_text(class_elem, "api_path", "")
            description = get_text(class_elem, "description", "")

            # Class overview (name, description, initialization)
            overview_parts = [f"Class: {name}"]
            if api_path:
                overview_parts.append(f"API Path: {api_path}")
            if description:
                overview_parts.append(f"Description: {description}")

            # Add initialization info to overview
            init_elem = class_elem.find("initialization")
            if init_elem is not None:
                params = get_text(init_elem, "parameters", "")
                init_desc = get_text(init_elem, "description", "")
                if params:
                    overview_parts.append(f"Initialization: __init__({params})")
                if init_desc:
                    overview_parts.append(f"Init Description: {init_desc}")

            classes.append("\n".join(overview_parts))
            examples.extend(get_examples(class_elem))

            # Extract methods
            methods_elem = class_elem.find("methods")
            if methods_elem is not None:
                for method_elem in methods_elem.findall("method"):
                    method_name = get_text(method_elem, "name", "unknown")
                    method_desc = get_text(method_elem, "description", "")
                    method_header = get_text(method_elem, "header", "")
                    method_output = get_text(method_elem, "output", "")

                    method_parts = [
                        f"Method: {name}.{method_name}",
                    ]
                    if method_header:
                        method_parts.append(f"Signature: {method_header}")
                    if method_desc:
                        method_parts.append(f"Description: {method_desc}")
                    if method_output:
                        method_parts.append(f"Output: {method_output}")

                    methods.append(
                        "\n".join(method_parts),
                    )
                    examples.extend(get_examples(method_elem))

        for function_elem in root.findall("function"):
            name = get_text(function_elem, "name", "Unknown")
            api_path = get_text(function_elem, "api_path", "")
            description = get_text(function_elem, "description", "")
            header = get_text(function_elem, "header", "")
            output = get_text(function_elem, "output", "")

            text_parts = [f"Function: {name}"]
            if api_path:
                text_parts.append(f"API Path: {api_path}")
            if header:
                text_parts.append(f"Signature: {header}")
            if description:
                text_parts.append(f"Description: {description}")
            if output:
                text_parts.append(f"Output: {output}")

            functions.append("\n".join(text_parts))
            examples.extend(get_examples(function_elem))

        return classes, methods, functions, examples

    except Exception as e:
        raise RuntimeError(f"Error extracting entities from api.xml: {str(e)}")
