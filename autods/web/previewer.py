import io
import os

import nbformat
import pandas as pd
import requests
import streamlit as st
from nbconvert import HTMLExporter
from PIL import Image


class ArtifactPreviewer:
    """Загрузка и визуализация артефактов (таблицы, текст, картинки, ноутбуки)."""

    TEXT_EXTENSIONS = {".txt", ".log", ".md", ".py", ".json", ".yaml", ".yml"}
    TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet", ".json"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    NB_EXTENSIONS = {".ipynb"}

    def __init__(
        self, api_base_url: str, max_bytes: int = 512 * 1024, table_rows: int = 200
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.max_bytes = max_bytes
        self.table_rows = table_rows
        if "artifact_preview_cache" not in st.session_state:
            st.session_state.artifact_preview_cache = {}

    @staticmethod
    def _extension(filename: str) -> str:
        return os.path.splitext(filename)[1].lower()

    def _cache_key(
        self, session_id: str, file_path: str, mtime: str | None = None
    ) -> str:
        return f"{session_id}::{file_path}::{mtime or ''}"

    def _fetch_file(
        self, session_id: str, file_path: str, *, allow_truncate: bool = True
    ) -> bytes:
        url = f"{self.api_base_url}/api/session/{session_id}/file"
        resp = requests.get(url, params={"file_path": file_path}, stream=True)
        resp.raise_for_status()
        if allow_truncate:
            return resp.raw.read(self.max_bytes + 1)
        return resp.content

    def preview(self, *, session_id: str, node: dict) -> dict:
        file_path = node["path"]
        mtime = node.get("mtime")
        key = self._cache_key(session_id, file_path, mtime)
        cache = st.session_state.artifact_preview_cache
        if key in cache:
            return cache[key]

        ext = self._extension(node["name"])
        allow_truncate = ext not in self.NB_EXTENSIONS  # ноутбуки загружаем целиком

        data = self._fetch_file(session_id, file_path, allow_truncate=allow_truncate)

        preview = {"type": "raw", "meta": node}
        try:
            if ext in self.NB_EXTENSIONS:
                preview = self._preview_notebook(data)
            elif ext in self.TABLE_EXTENSIONS:
                preview = self._preview_table(data, ext)
            elif ext in self.IMAGE_EXTENSIONS:
                preview = self._preview_image(data)
            elif ext in self.TEXT_EXTENSIONS or self._looks_like_text(data):
                preview = self._preview_text(data)
        except Exception as exc:
            preview = {"type": "error", "error": str(exc), "meta": node}

        cache[key] = preview
        return preview

    def _preview_text(self, data: bytes) -> dict:
        truncated = len(data) > self.max_bytes
        text = data[: self.max_bytes].decode("utf-8", errors="replace")
        return {"type": "text", "text": text, "truncated": truncated}

    def _preview_table(self, data: bytes, ext: str) -> dict:
        buf = io.BytesIO(data)
        if ext == ".parquet":
            df = pd.read_parquet(buf)
        elif ext in {".csv", ".tsv"}:
            sep = "," if ext == ".csv" else "\t"
            df = pd.read_csv(buf, sep=sep)
        else:  # .json
            df = pd.read_json(buf)

        truncated = len(df) > self.table_rows
        df_preview = df.head(self.table_rows)
        return {
            "type": "table",
            "dataframe": df_preview,
            "rows_total": len(df),
            "columns_total": len(df.columns),
            "truncated": truncated,
        }

    def _preview_image(self, data: bytes) -> dict:
        image = Image.open(io.BytesIO(data))
        return {
            "type": "image",
            "image": image,
            "format": image.format,
            "size": image.size,
        }

    def _preview_notebook(self, data: bytes) -> dict:
        notebook_str = data.decode("utf-8", errors="replace")
        nb = nbformat.reads(notebook_str, as_version=4)
        exporter = HTMLExporter()
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True
        html, _ = exporter.from_notebook_node(nb)
        return {"type": "notebook", "html": html}

    @staticmethod
    def _looks_like_text(data: bytes) -> bool:
        return not data or all(b >= 32 or b in (9, 10, 13) for b in data[:1024])
