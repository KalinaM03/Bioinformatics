from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.logging import setup_logging
from app import db as dbm
from app.services import scan_folder


setup_logging()

app = FastAPI(title="BioMed File Registry (MVP)", version="1.0.0")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = str(BASE_DIR / "web" / "templates")
STATIC_DIR = str(BASE_DIR / "web" / "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _startup():
    dbm.init_db()


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None


def file_to_out(rec: dbm.FileRecord) -> dbm.FileOut:
    meta = dbm.parse_metadata(rec)
    return dbm.FileOut(
        id=rec.id,
        path=rec.path,
        filename=rec.filename,
        extension=rec.extension,
        file_type=rec.file_type,
        size_bytes=rec.size_bytes,
        ctime_utc=rec.ctime_utc,
        mtime_utc=rec.mtime_utc,
        sha256=rec.sha256,
        hash_mode=rec.hash_mode,
        is_duplicate=rec.is_duplicate,
        project=rec.project,
        sample_id=rec.sample_id,
        patient_pseudo_id=rec.patient_pseudo_id,
        tags=[dbm.TagOut(name=t.name) for t in rec.tags],
        metadata=meta,
    )


def _build_url(base: str, params: dict[str, Any]) -> str:
    cleaned: dict[str, Any] = {}
    for k, v in params.items():
        if v is None or v == "" or v == []:
            continue
        if isinstance(v, bool):
            cleaned[k] = "true" if v else "false"
        else:
            cleaned[k] = v
    return base + ("?" + urlencode(cleaned, doseq=True) if cleaned else "")


def run_scan_job(job_id: str) -> None:
    db = dbm.SessionLocal()
    try:
        job = dbm.get_job(db, job_id)
        if not job:
            return
        dbm.set_job_status(db, job_id, "running")
        scan_folder(db, job_id, job.folder, job.hash_mode)
        dbm.set_job_status(db, job_id, "done")
    except Exception as e:
        dbm.set_job_status(db, job_id, "error", error_message=str(e))
    finally:
        db.close()


# --------------------------
# Web UI
# --------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(dbm.get_db)):
    rows = db.execute(
        select(dbm.FileRecord.file_type, func.count(), func.sum(dbm.FileRecord.size_bytes))
        .group_by(dbm.FileRecord.file_type)
        .order_by(func.count().desc())
    ).all()
    summary = [{"file_type": r[0], "count": int(r[1]), "total_size": int(r[2] or 0)} for r in rows]

    recent, _total = dbm.search_files(db, page=1, page_size=10)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": summary, "recent": [file_to_out(r) for r in recent]},
    )


@app.post("/scan")
def scan_form(
    background_tasks: BackgroundTasks,
    folder: str = Form(...),
    hash_mode: str = Form("quick"),
    db: Session = Depends(dbm.get_db),
):
    job = dbm.create_job(db, folder=folder, hash_mode=hash_mode)
    background_tasks.add_task(run_scan_job, job.id)
    return RedirectResponse(url="/jobs", status_code=303)


@app.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request, db: Session = Depends(dbm.get_db)):
    jobs = dbm.list_jobs(db, limit=50)
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})


@app.get("/files", response_class=HTMLResponse)
def files_page(
    request: Request,
    q: str | None = None,
    content_q: str | None = None,
    file_type: str | None = None,
    tags: str | None = None,
    project: str | None = None,
    sample_id: str | None = None,
    patient: str | None = None,
    min_mb: str | None = None,
    max_mb: str | None = None,
    after: str | None = None,
    before: str | None = None,
    duplicates_only: str | None = None,
    sort: str = "mtime_desc",
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(dbm.get_db),
):
    file_types = [ft.strip() for ft in (file_type or "").split(",") if ft.strip()] or None
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] or None

    def mb_to_bytes(x: str | None) -> int | None:
        if not x:
            return None
        try:
            return int(float(x) * 1024 * 1024)
        except Exception:
            return None

    min_size = mb_to_bytes(min_mb)
    max_size = mb_to_bytes(max_mb)
    dup = (duplicates_only or "false").lower() == "true"

    items, total = dbm.search_files(
        db,
        q=q,
        content_q=content_q,
        file_types=file_types,
        tags=tag_list,
        project=project or None,
        sample_id=sample_id or None,
        patient=patient or None,
        min_size=min_size,
        max_size=max_size,
        modified_after=_parse_dt(after),
        modified_before=_parse_dt(before),
        duplicates_only=dup,
        sort=sort,
        page=page,
        page_size=page_size,
    )
    pages = max(1, (total + page_size - 1) // page_size) if total else 1

    base_params = {
        "q": q or "",
        "content_q": content_q or "",
        "file_type": file_type or "",
        "tags": tags or "",
        "project": project or "",
        "sample_id": sample_id or "",
        "patient": patient or "",
        "min_mb": min_mb or "",
        "max_mb": max_mb or "",
        "after": after or "",
        "before": before or "",
        "duplicates_only": "true" if dup else "false",
        "sort": sort or "mtime_desc",
        "page_size": page_size,
    }

    prev_url = _build_url("/files", {**base_params, "page": max(1, page - 1)})
    next_url = _build_url("/files", {**base_params, "page": min(pages, page + 1)})

    export_csv_url = _build_url("/api/export", {**base_params, "format": "csv"})
    export_json_url = _build_url("/api/export", {**base_params, "format": "json"})

    return templates.TemplateResponse(
        "files.html",
        {
            "request": request,
            "items": [file_to_out(r) for r in items],
            "total": total,
            "page": page,
            "pages": pages,
            "q": q or "",
            "content_q": content_q or "",
            "file_type": file_type or "",
            "tags": tags or "",
            "project": project or "",
            "sample_id": sample_id or "",
            "patient": patient or "",
            "min_mb": min_mb or "",
            "max_mb": max_mb or "",
            "after": after or "",
            "before": before or "",
            "duplicates_only": dup,
            "sort": sort or "mtime_desc",
            "prev_url": prev_url,
            "next_url": next_url,
            "export_csv_url": export_csv_url,
            "export_json_url": export_json_url,
        },
    )


@app.get("/files/{file_id}", response_class=HTMLResponse)
def file_detail(request: Request, file_id: str, db: Session = Depends(dbm.get_db)):
    rec = db.get(dbm.FileRecord, file_id)
    if not rec:
        raise HTTPException(status_code=404, detail="File not found")
    out = file_to_out(rec).model_dump()
    out["updated_at_utc"] = rec.updated_at_utc
    return templates.TemplateResponse("file_detail.html", {"request": request, "file": out})


# --------------------------
# REST API
# --------------------------

@app.post("/api/scan", response_model=dbm.ScanResponse)
def api_scan(req: dbm.ScanRequest, background_tasks: BackgroundTasks, db: Session = Depends(dbm.get_db)):
    job = dbm.create_job(db, folder=req.folder, hash_mode=req.hash_mode)
    background_tasks.add_task(run_scan_job, job.id)
    return dbm.ScanResponse(job_id=job.id, status=job.status)


@app.get("/api/jobs", response_model=list[dbm.JobOut])
def api_jobs(db: Session = Depends(dbm.get_db), limit: int = 50):
    jobs = dbm.list_jobs(db, limit=limit)
    return [
        dbm.JobOut(
            id=j.id,
            folder=j.folder,
            hash_mode=j.hash_mode,
            status=j.status,
            started_at_utc=j.started_at_utc,
            finished_at_utc=j.finished_at_utc,
            files_seen=j.files_seen,
            files_indexed=j.files_indexed,
            files_updated=j.files_updated,
            files_failed=j.files_failed,
            error_message=j.error_message,
            created_at_utc=j.created_at_utc,
        )
        for j in jobs
    ]


@app.get("/api/files", response_model=dbm.FilesPage)
def api_files(
    q: str | None = None,
    content_q: str | None = None,
    file_type: str | None = None,
    tag: list[str] | None = Query(default=None),
    project: str | None = None,
    sample_id: str | None = None,
    patient: str | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    modified_after: str | None = None,
    modified_before: str | None = None,
    duplicates_only: bool = False,
    sort: str = "mtime_desc",
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(dbm.get_db),
):
    file_types = [ft.strip() for ft in (file_type or "").split(",") if ft.strip()] or None
    items, total = dbm.search_files(
        db,
        q=q,
        content_q=content_q,
        file_types=file_types,
        tags=tag,
        project=project,
        sample_id=sample_id,
        patient=patient,
        min_size=min_size,
        max_size=max_size,
        modified_after=_parse_dt(modified_after),
        modified_before=_parse_dt(modified_before),
        duplicates_only=duplicates_only,
        sort=sort,
        page=page,
        page_size=page_size,
    )
    return dbm.FilesPage(items=[file_to_out(r) for r in items], page=page, page_size=page_size, total=total)


@app.get("/api/files/{file_id}", response_model=dbm.FileOut)
def api_file(file_id: str, db: Session = Depends(dbm.get_db)):
    rec = db.get(dbm.FileRecord, file_id)
    if not rec:
        raise HTTPException(status_code=404, detail="File not found")
    return file_to_out(rec)


@app.post("/api/files/{file_id}/fields")
def api_update_fields(file_id: str, payload: dbm.UpdateFields, db: Session = Depends(dbm.get_db)):
    rec = db.get(dbm.FileRecord, file_id)
    if not rec:
        raise HTTPException(status_code=404, detail="File not found")
    if payload.project is not None:
        rec.project = payload.project or None
    if payload.sample_id is not None:
        rec.sample_id = payload.sample_id or None
    if payload.patient_pseudo_id is not None:
        rec.patient_pseudo_id = payload.patient_pseudo_id or None
    rec.updated_at_utc = datetime.utcnow()
    db.add(rec)
    db.commit()
    return {"ok": True}


@app.post("/api/files/{file_id}/tags")
def api_set_tags(file_id: str, tags: list[str], db: Session = Depends(dbm.get_db)):
    rec = db.get(dbm.FileRecord, file_id)
    if not rec:
        raise HTTPException(status_code=404, detail="File not found")
    dbm.set_file_tags(db, rec, tags)
    return {"ok": True, "tags": [t.name for t in rec.tags]}


@app.get("/api/export")
def api_export(
    format: str = "csv",
    q: str | None = None,
    content_q: str | None = None,
    file_type: str | None = None,
    tags: str | None = None,
    project: str | None = None,
    sample_id: str | None = None,
    patient: str | None = None,
    min_mb: str | None = None,
    max_mb: str | None = None,
    after: str | None = None,
    before: str | None = None,
    duplicates_only: str | None = None,
    sort: str = "mtime_desc",
    db: Session = Depends(dbm.get_db),
):
    file_types = [ft.strip() for ft in (file_type or "").split(",") if ft.strip()] or None
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] or None

    def mb_to_bytes(x: str | None) -> int | None:
        if not x:
            return None
        try:
            return int(float(x) * 1024 * 1024)
        except Exception:
            return None

    min_size = mb_to_bytes(min_mb)
    max_size = mb_to_bytes(max_mb)
    dup = (duplicates_only or "false").lower() == "true"

    items, _total = dbm.search_files(
        db,
        q=q,
        content_q=content_q,
        file_types=file_types,
        tags=tag_list,
        project=project or None,
        sample_id=sample_id or None,
        patient=patient or None,
        min_size=min_size,
        max_size=max_size,
        modified_after=_parse_dt(after),
        modified_before=_parse_dt(before),
        duplicates_only=dup,
        sort=sort,
        page=1,
        page_size=1_000_000,
    )

    rows: list[dict[str, Any]] = []
    for r in items:
        rows.append({
            "id": r.id,
            "path": r.path,
            "filename": r.filename,
            "file_type": r.file_type,
            "size_bytes": r.size_bytes,
            "mtime_utc": r.mtime_utc.isoformat(),
            "sha256": r.sha256 or "",
            "hash_mode": r.hash_mode,
            "is_duplicate": r.is_duplicate,
            "project": r.project or "",
            "sample_id": r.sample_id or "",
            "patient_pseudo_id": r.patient_pseudo_id or "",
            "tags": ",".join([t.name for t in r.tags]),
            "metadata_json": r.metadata_json or "",
        })

    fmt = (format or "csv").lower()
    if fmt == "json":
        return JSONResponse(rows)

    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    else:
        buf.write("\n")
    buf.seek(0)

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=biomed_registry_export.csv"},
    )
