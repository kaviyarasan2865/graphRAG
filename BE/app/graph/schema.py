"""Render the GSQL schema with the correct embedding dimension from config.

The static gsql/01_schema.gsql defaults to DIMENSION=1024 (bge-m3). When the
embedding provider changes (e.g. Gemini = 768), the dimension must match. This
helper reads the .gsql file and substitutes the DIMENSION so the schema can be
applied programmatically via the client.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import BASE_DIR, settings

SCHEMA_FILE = BASE_DIR / "gsql" / "01_schema.gsql"
QUERIES_FILE = BASE_DIR / "gsql" / "02_queries.gsql"

_DIM_RE = re.compile(r"(ADD VECTOR ATTRIBUTE emb\(DIMENSION=)\d+", re.I)


def render_schema(dim: int | None = None) -> str:
    dim = dim or settings.embed_dim
    text = Path(SCHEMA_FILE).read_text(encoding="utf-8")
    return _DIM_RE.sub(rf"\g<1>{dim}", text)


def read_queries() -> str:
    return Path(QUERIES_FILE).read_text(encoding="utf-8")
