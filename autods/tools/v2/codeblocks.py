from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from langchain_core.messages import HumanMessage
from langgraph.runtime import get_runtime

from autods.prompting.prompt_store import prompt_store
from autods.tools.base import BaseTool, ToolError
from autods.tools.v2._lark_codeblocks import parse_fenced_blocks
from autods.tools.v2.ipython import IPythonTool
from autods.tools.v2.shell import ShellTool
from autods.utils.parsers import parse_json

Lang = Literal["python", "bash"]


@dataclass
class CodeBlock:
    index: int
    lang: Lang
    code: str
    file: str | None = None


def parse_code_blocks(text: str) -> list[CodeBlock]:
    pairs = parse_fenced_blocks(text or "")
    return [
        CodeBlock(index=i, lang=lang, code=code)
        for i, (lang, code) in enumerate(pairs, start=1)
    ]


def _is_language_header(h: str) -> bool:
    """Check if header is a language keyword."""
    tag = (h or "").strip().lower()
    return tag in {"python", "py", "bash", "sh", "shell", "zsh"}


def _is_file_header(h: str) -> bool:
    """Check if header looks like a file path."""
    if not h or _is_language_header(h):
        return False
    # Accept headers that look like filesystem paths or filenames.
    return any(ch in h for ch in ("/", "\\", ".")) and not h.strip().endswith(":")


def _detect_file_path_comment(code: str) -> str | None:
    """Extract file path from first line comment in Python code.

    Returns the file path if first line is '# path/to/file.ext', else None.
    """
    lines = code.splitlines()
    if not lines:
        return None

    first_line = lines[0].strip()
    if not first_line.startswith("#"):
        return None

    # Extract text after '#' and strip whitespace
    path_candidate = first_line[1:].strip()

    # Validate as file path
    return path_candidate if _is_file_header(path_candidate) else None


def _validate_file_path(path_str: str, base_cwd: Path) -> Path:
    """Resolve and validate file path is within project directory."""
    path = Path(path_str)
    target = path.resolve() if path.is_absolute() else (base_cwd / path).resolve()

    try:
        target.relative_to(base_cwd)
    except ValueError:
        raise ToolError(f"Invalid path: {path_str} escapes project directory")

    return target


def _collect_human_text(msg: HumanMessage) -> str:
    """Extract a text representation from a HumanMessage (handles LC content lists)."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    # content may be a list of {type: "text"|"image_url", ...}
    parts = [
        str(item.get("text", ""))
        for item in (content or [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p)


def _get_base_cwd() -> Path:
    """Get project path from runtime context."""
    runtime = get_runtime()
    context = getattr(runtime, "context", None)
    return Path(getattr(context, "project_path", Path.cwd())) if context else Path.cwd()


def _get_relative_path(path: Path, base_cwd: Path) -> str:
    """Get relative path string for display."""
    return (
        str(path.relative_to(base_cwd)) if path.is_relative_to(base_cwd) else str(path)
    )


async def _handle_file_operation(blk: CodeBlock, base_cwd: Path) -> str | None:
    """Handle file creation operations from Python code blocks."""
    from autods.tools.v2._apply_patch_model import Add
    from autods.tools.v2.apply_patch import _apply_add

    if blk.file:
        file_path_str = blk.file
    else:
        detected_path = _detect_file_path_comment(blk.code)
        if not detected_path:
            return None
        file_path_str = detected_path
        lines = blk.code.splitlines()
        blk.code = "\n".join(lines[1:])  # Skip first line comment

    validated_path = _validate_file_path(file_path_str, base_cwd)
    header = f">>> [{blk.lang} #{blk.index}]"
    rel_path = _get_relative_path(validated_path, base_cwd)

    # Create new file
    add = Add(path=validated_path, content=blk.code)
    _apply_add(add)
    result = f"{header}\nFile operation: A {rel_path}"

    # Auto-execute Python or bash scripts
    exec_result = await _auto_execute_file(validated_path)
    if exec_result:
        result += f"\n{exec_result}"

    return result


async def _auto_execute_file(path: Path) -> str | None:
    """Auto-execute Python or bash script files."""
    sh = ShellTool()

    if path.suffix == ".py":
        command = f"python {path}"
    elif path.suffix in {".sh", ".bash"}:
        command = f"bash {path}"
    else:
        return None

    try:
        raw = await sh.execute(arg=command)
        # Handle both str and HumanMessage return types
        if isinstance(raw, str):
            payload = parse_json(raw) or {}
            output = str(payload.get("output", "")).rstrip()
        else:
            output = _collect_human_text(raw).rstrip()
        return f">>> [executed {path.name}]\n{output}".rstrip()
    except Exception as e:
        return f">>> [execution failed]\nERROR: {e}"


async def _execute_python_block(
    blk: CodeBlock, base_cwd: Path, timeout: float | None = None
) -> str:
    """Execute a Python code block (file operation or IPython execution)."""
    # Try file operation first
    # file_result = await _handle_file_operation(blk, base_cwd)
    # if file_result:
    #     return file_result

    # Normal IPython execution
    ipy = IPythonTool(timeout=timeout)
    header = f">>> [{blk.lang} #{blk.index}]"
    msg = await ipy.execute(arg=blk.code)
    text = _collect_human_text(msg)
    return f"{header}\n{text}".rstrip()


async def _execute_bash_block(
    blk: CodeBlock, timeout: float | None = None
) -> tuple[str, int]:
    """Execute a bash code block and return (output, exit_code)."""
    code = (blk.code or "").strip()
    if not code:
        return "", 0

    sh = ShellTool(timeout=timeout)
    header = f">>> [{blk.lang} #{blk.index}]"

    raw = await sh.execute(arg=code)

    # Handle both str and HumanMessage return types
    raw_str = raw if isinstance(raw, str) else _collect_human_text(raw)

    try:
        payload = parse_json(raw_str) or {}
        output = str(payload.get("output", "")).rstrip()
        meta = payload.get("metadata", {}) or {}
        exit_code = int(meta.get("exit_code", 0))
        return f"{header}\n{output}".rstrip(), exit_code
    except Exception:
        # Keep raw string if parsing fails unexpectedly
        return f"{header}\n{raw_str}", 1


async def run_blocks(
    blocks: Sequence[CodeBlock], timeout: float | None = None
) -> tuple[str, int]:
    """Execute blocks sequentially via v2 tools.

    Returns: (aggregated_output, last_status)
      - last_status: 0 for success; non-zero on first error encountered.
    Always stops on first error as requested.
    """
    if not blocks:
        raise ToolError("No supported code blocks found.")

    base_cwd = _get_base_cwd()
    output_parts: list[str] = []
    status = 0

    for blk in blocks:
        try:
            if blk.lang == "python":
                result = await _execute_python_block(blk, base_cwd, timeout)
                output_parts.append(result)
            else:  # bash
                result, exit_code = await _execute_bash_block(blk, timeout)
                output_parts.append(result)
                if exit_code != 0:
                    status = exit_code
                    break
        except Exception as e:
            status = 1
            header = f">>> [{blk.lang} #{blk.index}]"
            output_parts.append(f"{header}\nERROR: {e}")
            break

    return "\n\n".join(output_parts).rstrip(), status


async def run_message(text: str, timeout: float | None = None) -> str:
    blocks = parse_code_blocks(text)
    aggregated, _status = await run_blocks(blocks, timeout)
    return aggregated


class CodeBlocksTool(BaseTool):
    name: str = "CodeBlock"
    usage: str = '<CodeBlock lang="python">print("Hello, World!")</CodeBlock>'
    timeout: float | None = None
    python_executor: Literal["jupyter", "bash"] = "jupyter"

    def get_prompt(self) -> str:
        return prompt_store.load("tools/codeblocks.md")

    async def execute(self, **kwargs) -> str | HumanMessage:
        text = kwargs.get("arg")
        lang = kwargs.get("lang")
        code = kwargs.get("code")
        file = kwargs.get("file")

        if lang and code:
            blocks = [CodeBlock(index=1, lang=lang, code=code, file=file)]
        elif lang and text:
            blocks = [CodeBlock(index=1, lang=lang, code=text, file=file)]
        elif isinstance(text, str) and text.strip():
            blocks = parse_code_blocks(text)
            if len(blocks) > 2:
                raise ToolError(
                    "Only two code blocks are allowed per execution. "
                    "Split your work and run at most two blocks step by step."
                )
        else:
            raise ToolError(f"Not correct usage. Got: {text}, expected: {self.usage}")

        # Ensure runtime context exists for downstream tools
        runtime = get_runtime()
        if getattr(runtime, "context", None) is None:
            raise ToolError("No runtime context available for tool execution.")

        aggregated, status = await run_blocks(blocks, timeout=self.timeout)
        if status != 0:
            # Surface as error to follow the repository's tool error pattern
            raise ToolError(aggregated or "Execution failed.")
        return aggregated
