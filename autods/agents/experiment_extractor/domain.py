"""Domain models for the ExperimentExtractor agent."""

from typing import Any

from pydantic import ConfigDict, Field

from autods.agents.domain import BaseAgentState, BaseThinkActAgent
from autods.tools.v2.toolkit_v2 import Toolkit
from autods.utils.llm_client import LLMClient


class ExperimentExtractorState(BaseAgentState):
    """State for the ExperimentExtractor agent."""

    extracted_experiments: list[dict[str, Any]] = Field(default_factory=list)
    pdf_path: str = ""


class ExperimentExtractorContext(BaseThinkActAgent):
    """Context for the ExperimentExtractor agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_client: LLMClient
    toolkit: Toolkit
    pdf_path: str
    use_vision: bool = False
