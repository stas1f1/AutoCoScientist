"""ExperimentExtractor agent for extracting experiments from scientific papers."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph

from autods.agents.base import BaseAgent
from autods.agents.experiment_extractor.domain import (
    ExperimentExtractorContext,
    ExperimentExtractorState,
)
from autods.agents.think_act_agent import create_think_act_graph
from autods.prompting.prompt_generator import PromptGenerator
from autods.prompting.prompt_store import prompt_store
from autods.tools.base import BaseTool
from autods.tools.v2.pdf_parser import PDFParserTool
from autods.tools.v2.toolkit_v2 import Toolkit
from autods.utils.config import Config
from autods.utils.llm_client import LLMClient


class ExperimentExtractorPromptGenerator(PromptGenerator):
    """Prompt generator for the ExperimentExtractor agent."""

    def __init__(self, pdf_path: str, tools: list[BaseTool]) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.tools = tools

    @property
    def system_prompt(self) -> SystemMessage:
        rules = prompt_store.load("experiment_extractor.md")
        tool_guidance = "\n\n---\n\n".join([tool.get_prompt() for tool in self.tools])
        return SystemMessage(content="\n\n".join([rules, tool_guidance]))

    @property
    def user_prompt(self) -> HumanMessage:
        return HumanMessage(
            content=f"[PDF Path]\n{self.pdf_path}\n\n"
            f"Extract all experiments from this scientific paper."
        )


class ExperimentExtractorAgent(BaseAgent):
    """Agent for extracting structured experiment descriptions from scientific papers."""

    def __init__(
        self,
        app_config: Config,
        pdf_path: str,
        use_vision: bool = False,
    ):
        """Initialize the ExperimentExtractor agent.

        Args:
            app_config: Application configuration
            pdf_path: Path to the PDF file to process
            use_vision: Whether to use vision mode for PDF parsing (default: False)
        """
        super().__init__(app_config)

        # Get agent config or use defaults
        agent_config = app_config.agents.get("experiment_extractor")
        if agent_config:
            model = agent_config.model
            max_steps = agent_config.max_steps or 30
        else:
            model = "gpt-4o"
            max_steps = 30

        # Initialize LLM client
        self.llm_client = LLMClient(model)
        self.max_steps = max_steps
        self.pdf_path = pdf_path
        self.use_vision = use_vision

        # Create toolkit with PDF parser
        self.toolkit = self._create_toolkit()

        # Create context
        self.context = ExperimentExtractorContext(
            llm_client=self.llm_client,
            toolkit=self.toolkit,
            pdf_path=pdf_path,
            use_vision=use_vision,
        )

    def _create_toolkit(self) -> Toolkit:
        """Create toolkit with PDF parser tool."""
        pdf_parser = PDFParserTool(
            llm_client=self.llm_client,
            use_vision=self.use_vision,
        )
        return Toolkit(pdf_parser)

    def runnable(
        self, checkpointer: BaseCheckpointSaver[Any] | None = None
    ) -> CompiledStateGraph:
        """Return the compiled state graph for the agent.

        Args:
            checkpointer: Optional checkpointer for state persistence

        Returns:
            Compiled state graph
        """
        # Create prompt generator
        prompt_generator = ExperimentExtractorPromptGenerator(
            pdf_path=self.pdf_path,
            tools=self.toolkit.tools,
        )

        # Create think-act graph
        workflow = create_think_act_graph(
            prompt_generator=prompt_generator,
            toolkit=self.toolkit,
            state_type=ExperimentExtractorState,
            context_type=ExperimentExtractorContext,
            max_steps=self.max_steps,
            throw_history=True,
            next_node=END,
        )

        return workflow if checkpointer is None else workflow
