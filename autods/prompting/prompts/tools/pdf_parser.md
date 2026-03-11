# PDFParser

Extract structured experiment descriptions from scientific papers in PDF format.

Operations:
1. **parse** - Extract text or images from PDF file
2. **chunk** - Split extracted text into sections or pages
3. **extract_experiments** - Extract experiment data from a text chunk
4. **merge_experiments** - Consolidate overlapping experiments

## Parse Operation

Extracts text content from PDF (or images in vision mode).

Attributes:
- operation: "parse" (required)
- pdf_path: (required) path to the PDF file

Example:
```
<PDFParser
operation="parse"
pdf_path="/path/to/paper.pdf"
/>
```

Returns JSON with full_text and total_pages.

## Chunk Operation

Splits extracted text into manageable sections.

Attributes:
- operation: "chunk" (required)
- text: (required) the full text to chunk
- strategy: (optional) "section" (default) or "page"

Example:
```
<PDFParser
operation="chunk"
text="[full text from parse operation]"
strategy="section"
/>
```

Returns JSON with array of chunks, each containing id, title, and content.

## Extract Experiments Operation

Uses LLM to extract structured experiment data from a chunk.

Attributes:
- operation: "extract_experiments" (required)
- chunk_text: (required) text of the chunk to analyze
- chunk_id: (optional) identifier for the chunk

Example:
```
<PDFParser
operation="extract_experiments"
chunk_text="[section text]"
chunk_id="section_3"
/>
```

Returns JSON with experiments array containing experiment_id, title, objective, methods, parameters, results_summary, location_in_paper.

## Merge Experiments Operation

Consolidates experiments mentioned across multiple sections.

Attributes:
- operation: "merge_experiments" (required)
- experiments_json: (required) JSON array or string of experiments to merge

Example:
```
<PDFParser
operation="merge_experiments"
experiments_json='[{"experiment_id": "EXP-1", ...}, ...]'
/>
```

Returns JSON with merged_experiments array and statistics.

## Typical Workflow

1. Parse PDF to get full text
2. Chunk text into sections
3. Extract experiments from each chunk
4. Merge all extracted experiments
