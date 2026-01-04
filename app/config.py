from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    db_url: str = os.getenv("BIOMED_DB_URL", f"sqlite:///{PROJECT_ROOT / 'biomed_registry.db'}")

    # Safety limits for huge files (metadata only)
    max_text_bytes: int = int(os.getenv("BIOMED_MAX_TEXT_BYTES", "262144"))  # 256 KB
    max_header_lines: int = int(os.getenv("BIOMED_MAX_HEADER_LINES", "200"))

    # Full-text search (FTS5) over *limited* file content (text-based formats only)
    # NOTE: This does NOT load whole file. Only the first N bytes / lines are indexed.
    fts_enabled: bool = os.getenv("BIOMED_FTS_ENABLED", "true").lower() in ("1", "true", "yes")
    fts_max_bytes: int = int(os.getenv("BIOMED_FTS_MAX_BYTES", str(1024 * 1024)))  # 1 MB
    fts_fastq_max_lines: int = int(os.getenv("BIOMED_FTS_FASTQ_MAX_LINES", "6000"))

    # Quick hash chunk (begin + end)
    quick_hash_chunk: int = int(os.getenv("BIOMED_QUICK_HASH_CHUNK", str(1024 * 1024)))  # 1 MB

settings = Settings()
