import re
import shlex
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.runtime import get_runtime

from autods.environments.jupyter import JupyterExecutor
from autods.environments.sandbox import LocalSandboxAdapter
from autods.prompting.prompt_store import prompt_store
from autods.tools.base import BaseTool, Observation, ToolError

# Common import aliases used in data science
COMMON_IMPORT_ALIASES: dict[str, str] = {
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "sns": "import seaborn as sns",
    "tf": "import tensorflow as tf",
    "torch": "import torch",
    "sk": "import sklearn as sk",
    "sp": "import scipy as sp",
    "xgb": "import xgboost as xgb",
    "lgb": "import lightgbm as lgb",
    "cb": "import catboost as cb",
}


def _extract_module_name(full_path: str) -> str | None:
    """Extract clean module name from a Rope path that may include venv paths.

    Args:
        full_path: Full module path like 'data.foo.venv...site-packages.pandas.core'

    Returns:
        Clean module name like 'pandas.core' or None if not extractable
    """
    # Handle site-packages paths
    if "site-packages." in full_path:
        # Extract everything after site-packages.
        parts = full_path.split("site-packages.")
        if len(parts) > 1:
            return parts[-1]
    # Handle lib.pythonX.Y paths (standard library in venvs)
    if ".lib.python" in full_path:
        # Skip these as they're typically duplicates
        return None
    # Skip paths that look like local venv paths
    if ".venv" in full_path or "venv" in full_path.split("."):
        return None
    return full_path


def _suggest_imports_with_rope(name: str, project_path: Path) -> list[str]:
    """Use Rope AutoImport to find import suggestions for a name.

    Args:
        name: The name to search for (e.g., 'pd', 'DataFrame', 'numpy')
        project_path: Path to the project root for Rope context

    Returns:
        List of suggested import statements, e.g. ['from pandas import DataFrame']
    """
    # Check for common aliases first (np, pd, plt, etc.)
    if name in COMMON_IMPORT_ALIASES:
        return [COMMON_IMPORT_ALIASES[name]]

    try:
        from rope.base.project import Project
        from rope.contrib.autoimport.sqlite import AutoImport
    except ImportError:
        return []

    suggestions: list[str] = []
    seen_modules: set[str] = set()
    project = Project(str(project_path))
    try:
        autoimport = AutoImport(project, memory=True)
        try:
            autoimport.generate_modules_cache()
            # Use get_modules for exact name matches
            modules = autoimport.get_modules(name)
            for module_path in modules:
                clean_module = _extract_module_name(module_path)
                if clean_module and clean_module not in seen_modules:
                    seen_modules.add(clean_module)
                    suggestions.append(f"from {clean_module} import {name}")
                    if len(suggestions) >= 5:
                        break
        finally:
            autoimport.close()
    except Exception:
        # If Rope fails for any reason, return empty suggestions
        pass
    finally:
        project.close()

    return suggestions


def _format_import_suggestions(name: str, suggestions: list[str]) -> str:
    """Format import suggestions into a readable block."""
    if not suggestions:
        return ""
    lines = [f"\n\n--- Suggested imports for '{name}' ---"]
    for suggestion in suggestions:
        lines.append(suggestion)
    return "\n".join(lines)


async def parse_python_errors(
    message: str,
    sandbox: Optional[LocalSandboxAdapter] = None,
    project_path: Optional[Path] = None,
) -> str:
    """Parse Python errors from a message and suggest fixes.

    Handles:
    - TypeError/ValueError with unexpected keyword arguments (fetches help)
    - NameError: suggests imports for undefined names
    - ModuleNotFoundError: suggests correct module names
    - ImportError: suggests correct import names
    """
    message = message.strip()

    # Handle NameError: name 'X' is not defined
    name_error_match = re.search(r"NameError: name '([^']+)' is not defined", message)
    if name_error_match and project_path is not None:
        missing_name = name_error_match.group(1)
        suggestions = _suggest_imports_with_rope(missing_name, project_path)
        if suggestions:
            message += _format_import_suggestions(missing_name, suggestions)
        return message

    # Handle ModuleNotFoundError: No module named 'X'
    module_error_match = re.search(
        r"ModuleNotFoundError: No module named '([^']+)'", message
    )
    if module_error_match and project_path is not None:
        missing_module = module_error_match.group(1)
        # Search for similar module names
        suggestions = _suggest_imports_with_rope(missing_module, project_path)
        if suggestions:
            message += _format_import_suggestions(missing_module, suggestions)
        return message

    # Handle ImportError: cannot import name 'X' from 'Y'
    import_error_match = re.search(
        r"ImportError: cannot import name '([^']+)' from '([^']+)'", message
    )
    if import_error_match and project_path is not None:
        missing_name = import_error_match.group(1)
        # Search for the name across all modules
        suggestions = _suggest_imports_with_rope(missing_name, project_path)
        if suggestions:
            message += _format_import_suggestions(missing_name, suggestions)
        return message

    # Handle unexpected keyword argument errors
    if "got an unexpected keyword argument" in message:
        # Extract function name from error message
        # Pattern: TypeError: function_name() got an unexpected keyword argument 'arg'
        # Or: TypeError: Class.method_name() got an unexpected keyword argument 'arg'
        match = re.search(
            r"(?:TypeError|ValueError):\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\(\)\s+got an unexpected keyword argument",
            message,
        )
        if match and sandbox is not None and project_path is not None:
            function_name = match.group(1)
            try:
                # Execute help() via sandbox
                help_command = f"python3 -c {shlex.quote(f'help({function_name})')}"
                sandbox_result = await sandbox.run(
                    ["bash", "-lc", help_command],
                    cwd=project_path,
                )
                if sandbox_result.exit_code == 0 and sandbox_result.stdout.strip():
                    help_output = sandbox_result.stdout.strip()
                    message = f"{message}\n\n--- Help for {function_name}() ---\n{help_output}"
            except Exception:
                # If help() execution fails, return original message
                pass

    return message


async def _observation_to_content(observation: Observation) -> list[str | dict]:
    """Convert an observation into LangChain content blocks."""
    message = (observation.message or "").strip()

    content: list[str | dict] = []
    if message:
        content.append({"type": "text", "text": message})

    for image in observation.base64_images or []:
        content.append({"type": "image_url", "image_url": {"url": image}})

    if not content:
        content.append(
            {"type": "text", "text": "Notebook cell executed with no output."}
        )
    return content


def _get_context() -> Any:
    """Fetch the current agent context from the running LangGraph runtime."""
    runtime = get_runtime()
    context = getattr(runtime, "context", None)
    if context is None:
        raise ToolError(
            "Jupyter tool requires an agent context with workspace information."
        )
    return context


def _get_executor(context: Any) -> JupyterExecutor:
    """Fetch or create the executor stored on the agent context."""
    executor = getattr(context, "jupyter_executor", None)
    if executor is not None:
        return executor

    project_path = Path(getattr(context, "project_path", Path.cwd()))
    project_path.mkdir(parents=True, exist_ok=True)

    python_env = getattr(context, "python_env", None)
    executor = JupyterExecutor(workspace=project_path, python_env=python_env)
    try:
        context.jupyter_executor = executor  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        object.__setattr__(context, "jupyter_executor", executor)
    return executor


class IPythonTool(BaseTool):
    name: str = "jupyter"
    timeout: float | None = None

    def get_prompt(self) -> str:
        return prompt_store.load("tools/ipython.md")

    async def execute(self, **kwargs) -> HumanMessage:
        code = kwargs.get("arg")
        if not isinstance(code, str) or not code.strip():
            raise ToolError(
                "Parameter 'code' is required and must be a non-empty string."
            )

        context = _get_context()
        executor = _get_executor(context)
        observation = await executor.run(code=code, timeout=self.timeout)

        # Get sandbox and project_path from context for error enhancement
        sandbox = getattr(context, "sandbox", None)
        project_path = Path(getattr(context, "project_path", Path.cwd()))

        if not observation.is_success:
            error_message = observation.message or "Notebook execution failed."
            # Enhance error message with help() output if applicable
            enhanced_message = await parse_python_errors(
                error_message, sandbox, project_path
            )
            raise ToolError(enhanced_message)

        return HumanMessage(content=await _observation_to_content(observation))
