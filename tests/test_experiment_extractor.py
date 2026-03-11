"""Tests for the ExperimentExtractor agent."""

import pytest
from unittest.mock import MagicMock, create_autospec, patch

from autods.agents.experiment_extractor import (
    ExperimentExtractorAgent,
    ExperimentExtractorContext,
    ExperimentExtractorState,
)
from autods.tools.v2.pdf_parser import PDFParserTool
from autods.utils.config import Config
from autods.utils.llm_client import LLMClient


@pytest.fixture
def mock_config():
    """Create a mock config."""
    config = MagicMock(spec=Config)
    config.agents = {}
    return config


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client with proper spec."""
    # Use create_autospec to create a mock that satisfies type checking
    mock_llm = create_autospec(LLMClient, instance=True)
    mock_llm.ainvoke = MagicMock()
    return mock_llm


@pytest.fixture
def agent(mock_config, mock_llm_client):
    """Create an ExperimentExtractor agent for testing."""
    with patch("autods.agents.experiment_extractor.experiment_extractor.LLMClient", return_value=mock_llm_client):
        agent = ExperimentExtractorAgent(
            app_config=mock_config,
            pdf_path="/test/paper.pdf",
            use_vision=False,
        )
        return agent


def test_agent_initialization(agent):
    """Test that the agent initializes correctly."""
    assert agent.pdf_path == "/test/paper.pdf"
    assert agent.use_vision is False
    assert agent.max_steps == 30
    assert agent.toolkit is not None
    assert agent.context is not None


def test_context_properties(agent):
    """Test that the context has required properties."""
    assert isinstance(agent.context, ExperimentExtractorContext)
    assert agent.context.pdf_path == "/test/paper.pdf"
    assert agent.context.use_vision is False
    assert agent.context.toolkit is not None


def test_state_initialization():
    """Test that the state initializes correctly."""
    state = ExperimentExtractorState(
        pdf_path="/test/paper.pdf",
        extracted_experiments=[],
    )
    assert state.pdf_path == "/test/paper.pdf"
    assert state.extracted_experiments == []


def test_toolkit_creation(agent):
    """Test that the toolkit is created with PDFParser tool."""
    assert len(agent.toolkit.tools) == 1
    assert agent.toolkit.tools[0].name == "PDFParser"


@pytest.mark.asyncio
async def test_pdf_parser_tool_parse_operation():
    """Test PDF parser parse operation."""
    pytest.importorskip("fitz", reason="PyMuPDF (fitz) not installed")

    tool = PDFParserTool()

    # Mock fitz (PyMuPDF)
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Sample text from page"

    # Make the mock doc iterable by making it return a list of pages
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.close = MagicMock()

    # Patch fitz.open directly at the module level where it's imported
    with patch("fitz.open", return_value=mock_doc):
        result = await tool.execute(operation="parse", pdf_path="/test/paper.pdf")

    # Parse JSON result
    import json
    result_data = json.loads(result)

    assert "mode" in result_data
    assert result_data["mode"] == "text"
    assert "total_pages" in result_data
    assert result_data["total_pages"] == 1
    assert "full_text" in result_data
    assert "Sample text from page" in result_data["full_text"]


@pytest.mark.asyncio
async def test_pdf_parser_tool_chunk_operation():
    """Test PDF parser chunk operation."""
    tool = PDFParserTool()

    sample_text = """
    --- PAGE 1 ---
    1. Introduction
    This is the introduction section.

    2. Methods
    This is the methods section.
    """

    result = await tool.execute(
        operation="chunk",
        text=sample_text,
        strategy="section"
    )

    assert "chunks" in result
    assert "total_chunks" in result


def test_pdf_parser_tool_llm_client_optional():
    """Test that PDFParser can be created without LLM client."""
    tool = PDFParserTool()
    assert tool.llm_client is None
    assert tool.use_vision is False


def test_runnable_returns_compiled_graph(agent):
    """Test that runnable() returns a compiled graph."""
    runnable = agent.runnable()
    assert runnable is not None
    # CompiledStateGraph should have certain methods
    assert hasattr(runnable, "ainvoke")
