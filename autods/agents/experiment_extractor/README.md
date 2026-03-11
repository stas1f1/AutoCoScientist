# ExperimentExtractor Agent

An autonomous agent that extracts structured experiment descriptions from scientific papers in PDF format.

## Overview

The ExperimentExtractorAgent follows the established autods agent patterns, using a Think-Act loop to autonomously:
1. Parse PDF files (text or vision mode)
2. Chunk content into sections
3. Extract experiments from each section
4. Merge overlapping experiments mentioned across sections

## Architecture

```
ExperimentExtractorAgent (BaseAgent)
├── State: ExperimentExtractorState
│   ├── extracted_experiments: List[Dict]
│   └── pdf_path: str
└── Context: ExperimentExtractorContext
    ├── llm_client: LLMClient
    ├── toolkit: Toolkit (with PDFParserTool)
    ├── pdf_path: str
    └── use_vision: bool
```

## Usage

### Basic Example

```python
from autods.agents.experiment_extractor import ExperimentExtractorAgent
from autods.utils.config import Config
from langchain_core.messages import HumanMessage

# Load config
config = Config.load("config.yaml")

# Create agent
agent = ExperimentExtractorAgent(
    app_config=config,
    pdf_path="path/to/scientific_paper.pdf",
    use_vision=False  # Use text mode (faster, cheaper)
)

# Run agent
result = await agent.runnable().ainvoke(
    input={"messages": [HumanMessage(content="Extract all experiments")]},
    context=agent.context,
)

# Access extracted experiments
experiments = result.get("extracted_experiments", [])
```

### Example Output

```json
[
  {
    "experiment_id": "EXP-1",
    "title": "Effect of temperature on catalyst yield",
    "objective": "Determine optimal reaction temperature for maximum yield",
    "methods": "Catalyst heated at temperatures ranging from 200-400°C in controlled atmosphere",
    "parameters": {
      "temperature_range": "200-400°C",
      "duration": "2 hours",
      "pressure": "1 atm"
    },
    "results_summary": "Maximum yield of 85% observed at 350°C",
    "location_in_paper": "Section 3.2, Pages 5-7",
    "chunk_identifiers": ["section_3", "section_4"]
  }
]
```

## Configuration

Add to `config.yaml`:

```yaml
agents:
  experiment_extractor:
    model: "gpt-4o"  # or "claude-opus-4-5"
    max_steps: 30
    use_vision: false
```

## PDF Parser Tool

The agent uses the `PDFParserTool` with four operations:

### 1. Parse
Extract text or images from PDF.

```xml
<PDFParser operation="parse" pdf_path="/path/to/paper.pdf"/>
```

### 2. Chunk
Split extracted text into sections.

```xml
<PDFParser operation="chunk" text="[full text]" strategy="section"/>
```

Strategies:
- `section`: Detect headers using regex (default)
- `page`: Split by page markers

### 3. Extract Experiments
Use LLM to extract structured experiment data from a chunk.

```xml
<PDFParser
  operation="extract_experiments"
  chunk_text="[section text]"
  chunk_id="section_3"
/>
```

### 4. Merge Experiments
Consolidate experiments mentioned across multiple sections.

```xml
<PDFParser
  operation="merge_experiments"
  experiments_json='[{...}, {...}]'
/>
```

## Parsing Modes

### Text Mode (Default)
- Uses PyMuPDF (fitz) for fast text extraction
- Cost-effective
- Works well for standard PDFs with selectable text

### Vision Mode
- Converts pages to images
- Better for complex layouts, tables, figures
- Requires vision-capable LLM (higher cost)
- Enable with `use_vision=True`

## Experiment Schema

Each extracted experiment contains:

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id` | string | Unique identifier (EXP-1, EXP-2, etc.) |
| `title` | string | Concise description of what was tested |
| `objective` | string | Research question or hypothesis |
| `methods` | string | Experimental procedures, techniques, equipment |
| `parameters` | dict | Key variables (temperature, dosage, etc.) |
| `results_summary` | string | Main findings or outcomes |
| `location_in_paper` | string | Section/page reference |
| `chunk_identifiers` | list[string] | Chunks where experiment was found |

## Workflow

The agent follows this process:

1. **Parse PDF**: Extract full text from the paper
2. **Chunk Text**: Split by sections (Introduction, Methods, Results, etc.)
3. **Extract Experiments**: Process each chunk to identify experiments
4. **Merge Experiments**: Consolidate overlapping mentions

## Dependencies

- **PyMuPDF** (`pymupdf>=1.24.0`): PDF text extraction
- **json-repair**: Parse LLM JSON responses
- **langchain-text-splitters**: Semantic chunking (available)

## Files

```
autods/agents/experiment_extractor/
├── __init__.py                    # Package exports
├── domain.py                      # State and Context models
├── experiment_extractor.py        # Main agent implementation
└── README.md                      # This file

autods/tools/v2/
└── pdf_parser.py                  # PDF parsing tool

autods/prompting/prompts/
├── experiment_extractor.md        # Agent system prompt
└── tools/
    └── pdf_parser.md              # Tool documentation

tests/
└── test_experiment_extractor.py   # Unit tests

examples/
└── experiment_extractor_example.py # Usage example
```

## Testing

Run tests:

```bash
pytest tests/test_experiment_extractor.py -v
```

## Extension Points

Future enhancements could include:

- Table extraction from experiments
- Figure analysis for methodology details
- Citation linking
- Multi-paper batch processing
- Comparative experiment analysis
- Export to structured formats (CSV, JSON-LD, etc.)

## Design Rationale

**Simplicity Choices:**
- Single-stage Think-Act pattern (not multi-stage pipeline)
- Integrated tool with operations (not multiple separate tools)
- Text-first with vision toggle (cost-effective default)
- Section-based chunking (more semantic than page-based)
- LLM-based merging (handles complex overlaps better than rules)

**Minimal Code:**
- Reuses all existing infrastructure (LLMClient, Toolkit, ThinkActAgent)
- No new base classes or abstractions
- Leverages available libraries
- ~500 lines total across all new files
