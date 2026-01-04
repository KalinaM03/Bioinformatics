# BioMed File Registry (MVP)

A lightweight, fully functional tool for a **registry + indexer** of large biomedical files.  
Works **only with metadata** (does not upload files and does not read their full content).

## What it does
- Import (recursive scan) of a folder
- File type detection (FASTQ/FASTA/VCF/DICOM/CSV/TSV/JSON/XML/PDF/NIFTI/...)
- Extraction of **lightweight** metadata by type (headers/columns/DICOM tags without pixels)
- Indexing into SQLite (local)
- Search/filter/sort via UI and REST API
- Export (CSV/JSON)
- (New, optional): **full-text search in content** for text-based formats (SQLite FTS5)

## Getting started
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:
- UI: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs

## Notes
- Database: biomed_registry.db (in the project root)
- Indexing: from “Home” → “Index folder.”
- For DICOM, it uses pydicom with stop_before_pixels=True to avoid loading pixel data.

Full-text search indexes only a limited portion of files (first N bytes/lines)
to remain fast and safe for large files. Controlled via env vars:
BIOMED_FTS_ENABLED, BIOMED_FTS_MAX_BYTES, BIOMED_FTS_FASTQ_MAX_LINES.
