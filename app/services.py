from __future__ import annotations

import gzip
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pydicom
from sqlalchemy.orm import Session

from app.config import settings
from app import db as dbm


# -------------------------
# Hashing (duplicates)
# -------------------------

def sha256_full(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_quick(path: str, chunk_size: int) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        first = f.read(chunk_size)
        h.update(first)
        if size > chunk_size:
            try:
                f.seek(max(0, size - chunk_size))
                last = f.read(chunk_size)
                h.update(last)
            except Exception:
                pass
    h.update(str(size).encode("utf-8"))
    return h.hexdigest()


def compute_hash(path: str, mode: str) -> str | None:
    mode = (mode or "quick").lower()
    if mode == "none":
        return None
    if mode == "full":
        return sha256_full(path)
    return sha256_quick(path, settings.quick_hash_chunk)


# -------------------------
# File type detection
# -------------------------

EXT_MAP = {
    # Genomics
    ".fastq": "FASTQ",
    ".fq": "FASTQ",
    ".fasta": "FASTA",
    ".fa": "FASTA",
    ".fna": "FASTA",
    ".vcf": "VCF",
    ".bcf": "BCF",
    ".bam": "BAM",
    ".cram": "CRAM",
    ".sam": "SAM",
    ".bed": "BED",
    ".gff": "GFF",
    ".gtf": "GTF",
    # Imaging
    ".dcm": "DICOM",
    ".dicom": "DICOM",
    ".nii": "NIFTI",
    ".nii.gz": "NIFTI",
    # Tables & docs
    ".csv": "CSV",
    ".tsv": "TSV",
    ".txt": "TEXT",
    ".json": "JSON",
    ".xml": "XML",
    ".pdf": "PDF",
}

# Compressed common biomedical formats
COMPRESSED_SUFFIX_MAP = {
    ".fastq.gz": ("FASTQ", ".fastq.gz"),
    ".fq.gz": ("FASTQ", ".fq.gz"),
    ".fasta.gz": ("FASTA", ".fasta.gz"),
    ".fa.gz": ("FASTA", ".fa.gz"),
    ".fna.gz": ("FASTA", ".fna.gz"),
    ".vcf.gz": ("VCF", ".vcf.gz"),
    ".csv.gz": ("CSV", ".csv.gz"),
    ".tsv.gz": ("TSV", ".tsv.gz"),
    ".txt.gz": ("TEXT", ".txt.gz"),
    ".json.gz": ("JSON", ".json.gz"),
    ".xml.gz": ("XML", ".xml.gz"),
}


def detect_file_type(path: str) -> tuple[str, str]:
    p = Path(path)
    lower = p.name.lower()

    if lower.endswith(".nii.gz"):
        return "NIFTI", ".nii.gz"

    for suff, (ft, ext) in COMPRESSED_SUFFIX_MAP.items():
        if lower.endswith(suff):
            return ft, ext

    ext = p.suffix.lower()
    return EXT_MAP.get(ext, "OTHER"), ext


def is_dicom_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
        return len(pre) >= 132 and pre[128:132] == b"DICM"
    except Exception:
        return False


# -------------------------
# Metadata extractors (light)
# -------------------------

def _open_text_head(path: str, max_bytes: int) -> bytes:
    # Support .gz without reading whole file
    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read(max_bytes)
    with open(path, "rb") as f:
        return f.read(max_bytes)


def _read_text_head(path: str, max_bytes: int) -> str:
    raw = _open_text_head(path, max_bytes)
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc, errors="replace")
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_metadata(path: str, file_type: str) -> dict[str, Any] | None:
    file_type = (file_type or "OTHER").upper()
    try:
        if file_type == "VCF":
            return extract_vcf_metadata(path)
        if file_type in ("FASTA", "FASTQ"):
            return extract_fasta_fastq_metadata(path, file_type)
        if file_type in ("CSV", "TSV"):
            return extract_table_metadata(path, delimiter="," if file_type == "CSV" else "\t")
        if file_type == "JSON":
            return extract_json_metadata(path)
        if file_type == "XML":
            return extract_xml_metadata(path)
        if file_type == "DICOM":
            return extract_dicom_metadata(path)
        return None
    except Exception:
        return None


# -------------------------
# Content extraction for FTS (limited)
# -------------------------

# File types that are safe and useful to index as text (partial read).
TEXT_LIKE_TYPES = {
    "VCF",
    "FASTA",
    "FASTQ",
    "CSV",
    "TSV",
    "TEXT",
    "JSON",
    "XML",
    "BED",
    "GFF",
    "GTF",
    "SAM",
}


def _open_text_stream(path: str):
    """Open a file as a text stream (supports .gz)."""
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _extract_lines_with_prefix(path: str, prefix: str, max_lines: int, max_bytes: int) -> str:
    buf: list[str] = []
    total = 0
    try:
        with _open_text_stream(path) as f:
            for line in f:
                if line.startswith(prefix):
                    s = line.strip()
                    if not s:
                        continue
                    # Truncate absurdly long lines (e.g. FASTA descriptions).
                    if len(s) > 5000:
                        s = s[:5000]
                    buf.append(s)
                    total += len(s) + 1
                    if len(buf) >= max_lines or total >= max_bytes:
                        break
        return "\n".join(buf)[:max_bytes]
    except Exception:
        return ""


def extract_content_for_fts(path: str, file_type: str) -> str | None:
    """Return limited, indexable text for full-text search.

    Important: We intentionally do NOT read whole file.
    - For FASTA/FASTQ we index header lines only.
    - For other text-like formats we index first N bytes.
    """
    if not settings.fts_enabled:
        return None
    ft = (file_type or "OTHER").upper()
    if ft not in TEXT_LIKE_TYPES:
        return None

    # Sequence formats: keep only headers for signal-to-noise.
    if ft == "FASTA":
        return _extract_lines_with_prefix(path, ">", max_lines=4000, max_bytes=settings.fts_max_bytes)
    if ft == "FASTQ":
        return _extract_lines_with_prefix(
            path,
            "@",
            max_lines=settings.fts_fastq_max_lines,
            max_bytes=settings.fts_max_bytes,
        )

    # Other text-like formats: read first N bytes and decode.
    text = _read_text_head(path, settings.fts_max_bytes)
    # Trim to keep DB small.
    return text[: settings.fts_max_bytes] if text else None


def extract_vcf_metadata(path: str) -> dict[str, Any]:
    text = _read_text_head(path, settings.max_text_bytes)
    lines = text.splitlines()
    meta: dict[str, Any] = {"format": "VCF"}
    headers = 0
    for line in lines[: settings.max_header_lines]:
        if line.startswith("##"):
            headers += 1
            if line.startswith("##reference="):
                meta["reference"] = line.split("=", 1)[1].strip()
        elif line.startswith("#CHROM"):
            meta["columns"] = line.lstrip("#").split("\t")
            break
    meta["header_lines"] = headers
    return meta


def extract_fasta_fastq_metadata(path: str, file_type: str) -> dict[str, Any]:
    text = _read_text_head(path, 8192)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    meta: dict[str, Any] = {"format": file_type}
    if not lines:
        return meta
    first = lines[0]
    if file_type == "FASTA" and first.startswith(">"):
        meta["first_record_id"] = first[1:].split()[0]
    if file_type == "FASTQ" and first.startswith("@"):
        meta["first_record_id"] = first[1:].split()[0]
    return meta


def extract_table_metadata(path: str, delimiter: str) -> dict[str, Any]:
    text = _read_text_head(path, 65536)
    lines = [l for l in text.splitlines() if l.strip()]
    meta: dict[str, Any] = {"format": "TABLE"}
    if not lines:
        return meta
    cols = lines[0].split(delimiter)
    meta["columns_count"] = len(cols)
    meta["columns_preview"] = cols[:50]
    return meta


def extract_json_metadata(path: str) -> dict[str, Any]:
    text = _read_text_head(path, settings.max_text_bytes)
    meta: dict[str, Any] = {"format": "JSON"}
    try:
        import json
        obj = json.loads(text)
        if isinstance(obj, dict):
            meta["top_level_keys"] = list(obj.keys())[:50]
        elif isinstance(obj, list):
            meta["top_level_type"] = "list"
            meta["list_length_preview"] = min(len(obj), 1_000_000)
        else:
            meta["top_level_type"] = type(obj).__name__
    except Exception:
        meta["parse_error"] = True
    return meta


def extract_xml_metadata(path: str) -> dict[str, Any]:
    text = _read_text_head(path, settings.max_text_bytes)
    meta: dict[str, Any] = {"format": "XML"}
    for line in text.splitlines()[:50]:
        line = line.strip()
        if line.startswith("<") and not line.startswith("<?") and not line.startswith("<!"):
            tag = line[1:].split()[0].split(">")[0].strip("/")
            if tag:
                meta["root_tag_guess"] = tag[:120]
            break
    return meta


def extract_dicom_metadata(path: str) -> dict[str, Any] | None:
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "Modality",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "PatientID",
                "StudyDate",
            ],
        )
    except Exception:
        if not is_dicom_file(path):
            return None
        return {"format": "DICOM", "read_error": True}

    def safe_get(tag: str) -> str | None:
        try:
            v = getattr(ds, tag, None)
            return str(v)[:200] if v is not None else None
        except Exception:
            return None

    return {
        "format": "DICOM",
        "modality": safe_get("Modality"),
        "study_uid": safe_get("StudyInstanceUID"),
        "series_uid": safe_get("SeriesInstanceUID"),
        "sop_uid": safe_get("SOPInstanceUID"),
        "patient_id": safe_get("PatientID"),
        "study_date": safe_get("StudyDate"),
    }


# -------------------------
# Ingestion / scanning
# -------------------------

def _dt_utc_from_ts(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def scan_folder(db: Session, job_id: str, folder: str, hash_mode: str) -> None:
    folder = str(folder)
    hash_mode = (hash_mode or "quick").lower()

    for dirpath, _dirnames, filenames in os.walk(folder):
        for fn in filenames:
            full_path = os.path.join(dirpath, fn)
            try:
                dbm.bump_job_counters(db, job_id, files_seen=1)

                st = os.stat(full_path)
                size = int(st.st_size)
                ctime = _dt_utc_from_ts(st.st_ctime)
                mtime = _dt_utc_from_ts(st.st_mtime)

                file_type, ext = detect_file_type(full_path)
                if file_type == "OTHER" and is_dicom_file(full_path):
                    file_type = "DICOM"

                digest = compute_hash(full_path, hash_mode)
                is_dup = dbm.find_duplicate_by_hash(db, digest, hash_mode) if digest else False

                meta = extract_metadata(full_path, file_type)

                _rec, action = dbm.upsert_file(
                    db,
                    path=os.path.abspath(full_path),
                    filename=fn,
                    extension=ext,
                    file_type=file_type,
                    size_bytes=size,
                    ctime_utc=ctime,
                    mtime_utc=mtime,
                    sha256=digest,
                    hash_mode=hash_mode,
                    is_duplicate=is_dup,
                    metadata=meta,
                )

                if action == "inserted":
                    dbm.bump_job_counters(db, job_id, files_indexed=1)
                elif action == "updated":
                    dbm.bump_job_counters(db, job_id, files_updated=1)

                # Optional: Full-text content index (SQLite FTS5).
                # We never read whole files. Only small, safe portions are indexed.
                try:
                    if settings.fts_enabled and (file_type or "").upper() in TEXT_LIKE_TYPES:
                        if action in ("inserted", "updated") or not dbm.fts_has_entry(db, _rec.id):
                            content = extract_content_for_fts(full_path, file_type)
                            dbm.fts_upsert(db, _rec.id, content)
                except Exception:
                    # If FTS is unavailable, the app still functions as a metadata registry.
                    pass

            except Exception:
                dbm.bump_job_counters(db, job_id, files_failed=1)
                continue
