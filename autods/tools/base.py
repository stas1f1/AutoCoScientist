from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


class ToolError(Exception):
    """Base class for tool errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message: str = message


class Observation(BaseModel):
    """Represents the result of a tool execution."""

    is_success: bool = Field(default=True)
    message: str = Field(default="")
    base64_images: Optional[List[str]] = Field(default=None)


class BaseToolCall(BaseModel):
    name: str
    params: Dict[str, Any]


class BaseTool(ABC, BaseModel):
    name: str
    usage: str = Field(default="", description="Usage example of the tool.")

    @abstractmethod
    def get_prompt(self) -> str:
        """Return the prompt of the tool."""

    def success_response(self, result: str) -> HumanMessage:
        """Return the response of the tool."""
        return HumanMessage(content=result, role="tool")

    def error_response(self, error: str) -> HumanMessage:
        """Return the error response of the tool."""
        return HumanMessage(
            content=(
                f"[ERROR] FIX and TRY again\n"
                f"{error}\n"
                "[DEBUG OPTIONS]: \n"
                "A. Python `help()`"
            ),
            role="tool",
        )

    async def __call__(
        self,
        **kwargs,
    ) -> Any:
        """Execute the tool with given parameters. With error handling."""
        try:
            message = await self.execute(**kwargs)
            if isinstance(message, str):
                return self.success_response(message)
            elif isinstance(message, HumanMessage):
                return message
        except Exception as e:
            return self.error_response(str(e))

    @abstractmethod
    async def execute(self, **kwargs) -> str | HumanMessage:
        """Execute the tool with given parameters."""

    def basetool2langgraph(self) -> Any:
        """Convert BaseTool to LangChain StructuredTool for use with LangGraph."""

        from langchain.tools import StructuredTool
        from pydantic import BaseModel, ConfigDict, Field

        # Create a generic input schema that accepts any keyword arguments
        class ToolInput(BaseModel):
            model_config = ConfigDict(extra="allow", protected_namespaces=())

            # Default field to ensure schema is not empty
            kwargs: Dict[str, Any] = Field(
                default_factory=dict, description="Tool parameters"
            )

        # Create a wrapper function that passes through all arguments
        async def tool_wrapper(**kwargs: Any) -> str:
            result = await self(**kwargs)
            # LangChain expects a string or dict, extract content from HumanMessage
            if isinstance(result, HumanMessage):
                content = result.content
                if isinstance(content, str):
                    return content
                else:
                    return str(content)
            return str(result)

        tool_wrapper.__name__ = self.name

        return StructuredTool.from_function(
            coroutine=tool_wrapper,
            name=self.name,
            description=self.get_prompt(),
            args_schema=ToolInput,
        )

    @staticmethod
    def langgraph2basetool(langchain_tool: Any) -> BaseTool:
        """Convert LangChain tool back to BaseTool format."""

        class WrappedTool(BaseTool):
            """Wrapper that converts a LangChain tool to our BaseTool interface."""

            name: str = langchain_tool.name

            def get_prompt(self) -> str:
                return getattr(langchain_tool, "description", "")

            async def execute(self, **kwargs: Any) -> str | HumanMessage:
                if hasattr(langchain_tool, "ainvoke"):
                    result = await langchain_tool.ainvoke(kwargs)
                elif hasattr(langchain_tool, "arun"):
                    result = await langchain_tool.arun(**kwargs)
                else:
                    # Fallback to sync version
                    import asyncio

                    result = await asyncio.to_thread(langchain_tool.invoke, kwargs)
                return str(result)

        return WrappedTool()
