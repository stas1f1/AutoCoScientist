"""PDF Parser tool for extracting experiments from scientific papers."""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from pydantic import ConfigDict, Field

from autods.prompting.prompt_store import prompt_store
from autods.tools.base import BaseTool, ToolError
from autods.utils.llm_client import LLMClient


class PDFParserTool(BaseTool):
    """Tool for parsing PDF files and extracting structured experiment data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "PDFParser"
    llm_client: LLMClient | None = Field(default=None)
    use_vision: bool = Field(default=False)

    def get_prompt(self) -> str:
        return prompt_store.load("tools/pdf_parser.md")

    async def execute(self, **kwargs) -> str:
        operation = kwargs.get("operation")
        if not operation:
            raise ToolError("Parameter 'operation' is required")

        if operation == "parse":
            return await self._parse_pdf(kwargs)
        elif operation == "chunk":
            return await self._chunk_text(kwargs)
        elif operation == "extract_experiments":
            return await self._extract_experiments(kwargs)
        elif operation == "merge_experiments":
            return await self._merge_experiments(kwargs)
        else:
            raise ToolError(
                f"Unknown operation '{operation}'. "
                "Valid operations: parse, chunk, extract_experiments, merge_experiments"
            )

    async def _parse_pdf(self, kwargs: dict[str, Any]) -> str:
        """Extract text or images from PDF."""
        pdf_path = kwargs.get("pdf_path")
        if not pdf_path:
            raise ToolError("Parameter 'pdf_path' is required for parse operation")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ToolError(
                "PyMuPDF is not installed. Install it with: pip install PyMuPDF>=1.24.0"
            )

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise ToolError(f"Failed to open PDF file: {e}")

        if self.use_vision:
            # Vision mode: convert pages to base64 images
            images = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for quality
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                images.append(
                    {
                        "page": page_num + 1,
                        "image": img_base64,
                    }
                )

            doc.close()
            return json.dumps(
                {
                    "mode": "vision",
                    "total_pages": len(images),
                    "message": f"Extracted {len(images)} pages as images. Use vision-capable LLM for analysis.",
                },
                indent=2,
            )
        else:
            # Text mode: extract text
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages.append({"page": page_num + 1, "text": text})

            doc.close()

            # Combine all text with page markers
            full_text = ""
            for page_data in pages:
                full_text += f"\n\n--- PAGE {page_data['page']} ---\n\n"
                full_text += page_data["text"]

            return json.dumps(
                {
                    "mode": "text",
                    "total_pages": len(pages),
                    "full_text": full_text.strip(),
                },
                indent=2,
            )

    async def _chunk_text(self, kwargs: dict[str, Any]) -> str:
        """Split text into sections based on headers."""
        text = kwargs.get("text")
        strategy = kwargs.get("strategy", "section")

        if not text:
            raise ToolError("Parameter 'text' is required for chunk operation")

        if strategy == "section":
            sections = self._chunk_by_sections(text)
        elif strategy == "page":
            sections = self._chunk_by_pages(text)
        else:
            raise ToolError(
                f"Unknown chunking strategy '{strategy}'. Valid: section, page"
            )

        return json.dumps(
            {
                "strategy": strategy,
                "total_chunks": len(sections),
                "chunks": sections,
            },
            indent=2,
        )

    def _chunk_by_sections(self, text: str) -> list[dict[str, Any]]:
        """Chunk text by detecting section headers."""
        sections = []

        # Pattern to detect section headers (e.g., "1. Introduction", "2.1 Methods")
        header_pattern = re.compile(
            r"^(\d+\.?\d*)\s+([A-Z][^\n]+)", re.MULTILINE
        )

        lines = text.split("\n")
        current_section = {
            "id": "section_0",
            "title": "Preamble",
            "content": "",
            "header_number": "0",
        }

        for line in lines:
            match = header_pattern.match(line)
            if match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)

                # Start new section
                section_num = match.group(1)
                section_title = match.group(2).strip()
                current_section = {
                    "id": f"section_{len(sections)}",
                    "title": section_title,
                    "content": line + "\n",
                    "header_number": section_num,
                }
            else:
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)

        # If no sections detected, fall back to page-based chunking
        if len(sections) <= 1:
            return self._chunk_by_pages(text)

        return sections

    def _chunk_by_pages(self, text: str) -> list[dict[str, Any]]:
        """Chunk text by page markers."""
        page_pattern = re.compile(r"--- PAGE (\d+) ---")
        pages = []

        parts = page_pattern.split(text)

        # First part is before any page marker
        if parts[0].strip():
            pages.append(
                {
                    "id": "page_0",
                    "title": "Preamble",
                    "content": parts[0].strip(),
                    "page_number": "0",
                }
            )

        # Process page pairs (page_num, content)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                page_num = parts[i]
                content = parts[i + 1].strip()
                if content:
                    pages.append(
                        {
                            "id": f"page_{page_num}",
                            "title": f"Page {page_num}",
                            "content": content,
                            "page_number": page_num,
                        }
                    )

        return pages

    async def _extract_experiments(self, kwargs: dict[str, Any]) -> str:
        """Extract experiments from a text chunk using LLM."""
        chunk_text = kwargs.get("chunk_text")
        chunk_id = kwargs.get("chunk_id", "unknown")

        if not chunk_text:
            raise ToolError(
                "Parameter 'chunk_text' is required for extract_experiments operation"
            )

        if not self.llm_client:
            raise ToolError("LLM client not configured for PDFParser tool")

        prompt = f"""Extract all experiments described in this text section from a scientific paper.

TEXT SECTION:
{chunk_text}

For each experiment you find, extract the following information:
- experiment_id: A unique identifier (e.g., "EXP-1", "EXP-2")
- title: A concise title describing the experiment
- objective: What the experiment aims to test or measure
- methods: How the experiment was conducted (procedures, techniques)
- parameters: Key experimental parameters (e.g., temperature, pressure, concentrations)
- results_summary: Brief summary of reported results
- location_in_paper: Where this experiment is described (section name/number)

Return your response as a JSON array of experiments. If no experiments are found, return an empty array.

Example format:
[
  {{
    "experiment_id": "EXP-1",
    "title": "Effect of temperature on catalyst yield",
    "objective": "Determine optimal temperature for maximum yield",
    "methods": "Heated catalyst at varying temperatures (200-400°C) for 2 hours",
    "parameters": {{"temperature": "200-400°C", "duration": "2 hours"}},
    "results_summary": "Maximum yield of 85% observed at 350°C",
    "location_in_paper": "Section 3.2"
  }}
]

Return ONLY the JSON array, no other text.
"""

        try:
            response = await self.llm_client.ainvoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Try to parse JSON, with repair if needed
            try:
                import json_repair

                experiments = json_repair.loads(response_text)
            except Exception:
                # Fallback to standard json
                import json

                # Try to extract JSON array from response
                json_match = re.search(
                    r"\[.*\]", response_text, re.DOTALL
                )
                if json_match:
                    experiments = json.loads(json_match.group(0))
                else:
                    experiments = []

            # Add chunk_id to each experiment
            for exp in experiments:
                exp["chunk_id"] = chunk_id

            return json.dumps(
                {
                    "chunk_id": chunk_id,
                    "experiments_found": len(experiments),
                    "experiments": experiments,
                },
                indent=2,
            )

        except Exception as e:
            raise ToolError(f"Failed to extract experiments: {e}")

    async def _merge_experiments(self, kwargs: dict[str, Any]) -> str:
        """Merge overlapping experiments from multiple chunks using LLM."""
        experiments_json = kwargs.get("experiments_json")

        if not experiments_json:
            raise ToolError(
                "Parameter 'experiments_json' is required for merge_experiments operation"
            )

        if not self.llm_client:
            raise ToolError("LLM client not configured for PDFParser tool")

        # Parse experiments if provided as string
        if isinstance(experiments_json, str):
            try:
                import json_repair

                experiments = json_repair.loads(experiments_json)
            except Exception:
                import json

                experiments = json.loads(experiments_json)
        else:
            experiments = experiments_json

        if not experiments or len(experiments) == 0:
            return json.dumps(
                {"merged_experiments": [], "total_merged": 0}, indent=2
            )

        prompt = f"""You are given a list of experiments extracted from different sections of a scientific paper.
Some experiments may be mentioned across multiple sections (e.g., described in Methods and results in Results).
Your task is to merge experiments that refer to the same experimental work.

EXPERIMENTS TO MERGE:
{json.dumps(experiments, indent=2)}

Instructions:
1. Identify experiments that describe the same experimental work (look for matching objectives, methods, parameters)
2. Merge information from all mentions of the same experiment
3. Combine chunk_id values into a list for experiments found across multiple sections
4. Preserve all unique information from each mention
5. Keep experiments separate if they are clearly different experimental procedures

Return a JSON array of merged experiments. Each experiment should have:
- experiment_id: Keep or assign a unique ID
- title: Consolidated title
- objective: Complete objective description
- methods: Complete methods description
- parameters: All parameters merged
- results_summary: Complete results summary
- location_in_paper: Where this experiment is described
- chunk_identifiers: Array of chunk_ids where this experiment was found

Return ONLY the JSON array, no other text.
"""

        try:
            response = await self.llm_client.ainvoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Try to parse JSON
            try:
                import json_repair

                merged = json_repair.loads(response_text)
            except Exception:
                import json

                # Try to extract JSON array
                json_match = re.search(
                    r"\[.*\]", response_text, re.DOTALL
                )
                if json_match:
                    merged = json.loads(json_match.group(0))
                else:
                    # If merging fails, return original experiments
                    merged = experiments

            return json.dumps(
                {
                    "merged_experiments": merged,
                    "total_merged": len(merged),
                    "original_count": len(experiments),
                },
                indent=2,
            )

        except Exception as e:
            raise ToolError(f"Failed to merge experiments: {e}")
