from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
    or_,
    and_,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import settings


# -------------------------
# SQLAlchemy setup
# -------------------------

class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.db_url,
    connect_args={"check_same_thread": False} if settings.db_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

    # SQLite full-text search virtual table (FTS5).
    # Used to search inside the *limited* textual content of supported file types.
    if settings.db_url.startswith("sqlite") and settings.fts_enabled:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS file_content_fts USING fts5("
                    "file_id UNINDEXED, content)"
                )
        except Exception:
            # If FTS5 is not available in the SQLite build, the app still works (metadata-only).
            pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------
# Models
# -------------------------

class FileTag(Base):
    __tablename__ = "file_tags"
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"), primary_key=True)
    tag_id: Mapped[str] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)


class Tag(Base):
    __tablename__ = "tags"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(80), unique=True, index=True)

    files: Mapped[list["FileRecord"]] = relationship(
        back_populates="tags",
        secondary="file_tags",
        lazy="selectin",
    )


class FileRecord(Base):
    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    path: Mapped[str] = mapped_column(Text, unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(512), index=True)
    extension: Mapped[str] = mapped_column(String(32), index=True)
    file_type: Mapped[str] = mapped_column(String(64), index=True)

    size_bytes: Mapped[int] = mapped_column(Integer, index=True)
    ctime_utc: Mapped[datetime] = mapped_column(DateTime, index=True)
    mtime_utc: Mapped[datetime] = mapped_column(DateTime, index=True)

    sha256: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    hash_mode: Mapped[str] = mapped_column(String(16), default="quick")  # none|quick|full
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Human-entered fields (safe / pseudo)
    project: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    sample_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    patient_pseudo_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # Extracted metadata stored as JSON string
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at_utc: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    tags: Mapped[list[Tag]] = relationship(
        back_populates="files",
        secondary="file_tags",
        lazy="selectin",
    )


class ScanJob(Base):
    __tablename__ = "scan_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    folder: Mapped[str] = mapped_column(Text, index=True)
    hash_mode: Mapped[str] = mapped_column(String(16), default="quick")
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|error

    started_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    files_seen: Mapped[int] = mapped_column(Integer, default=0)
    files_indexed: Mapped[int] = mapped_column(Integer, default=0)
    files_updated: Mapped[int] = mapped_column(Integer, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, default=0)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


Index("ix_files_type_size_mtime", FileRecord.file_type, FileRecord.size_bytes, FileRecord.mtime_utc)
Index("ix_files_hash", FileRecord.sha256, FileRecord.hash_mode)


# -------------------------
# Schemas (API)
# -------------------------

class ScanRequest(BaseModel):
    folder: str
    hash_mode: str = Field(default="quick", pattern="^(none|quick|full)$")


class ScanResponse(BaseModel):
    job_id: str
    status: str


class TagOut(BaseModel):
    name: str


class FileOut(BaseModel):
    id: str
    path: str
    filename: str
    extension: str
    file_type: str
    size_bytes: int
    ctime_utc: datetime
    mtime_utc: datetime
    sha256: str | None
    hash_mode: str
    is_duplicate: bool
    project: str | None
    sample_id: str | None
    patient_pseudo_id: str | None
    tags: list[TagOut] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class FilesPage(BaseModel):
    items: list[FileOut]
    page: int
    page_size: int
    total: int


class JobOut(BaseModel):
    id: str
    folder: str
    hash_mode: str
    status: str
    started_at_utc: datetime | None
    finished_at_utc: datetime | None
    files_seen: int
    files_indexed: int
    files_updated: int
    files_failed: int
    error_message: str | None
    created_at_utc: datetime


class UpdateFields(BaseModel):
    project: str | None = None
    sample_id: str | None = None
    patient_pseudo_id: str | None = None


# -------------------------
# CRUD helpers
# -------------------------

def _now() -> datetime:
    return datetime.utcnow()


def parse_metadata(rec: FileRecord) -> dict[str, Any] | None:
    if not rec.metadata_json:
        return None
    try:
        return json.loads(rec.metadata_json)
    except Exception:
        return None


# -------------------------
# FTS helpers (SQLite FTS5)
# -------------------------

def _fts_enabled() -> bool:
    return settings.db_url.startswith("sqlite") and settings.fts_enabled


def fts_has_entry(db: Session, file_id: str) -> bool:
    if not _fts_enabled():
        return False
    try:
        row = db.execute(
            text("SELECT 1 FROM file_content_fts WHERE file_id = :id LIMIT 1"),
            {"id": file_id},
        ).first()
        return row is not None
    except Exception:
        return False


def fts_upsert(db: Session, file_id: str, content: str | None) -> None:
    """Insert/update the content index for one file.

    content is expected to be *limited* already (first N bytes/lines).
    """
    if not _fts_enabled():
        return
    try:
        # Keep the index small and robust.
        txt = (content or "").strip()
        db.execute(text("DELETE FROM file_content_fts WHERE file_id = :id"), {"id": file_id})
        if txt:
            db.execute(
                text("INSERT INTO file_content_fts(file_id, content) VALUES(:id, :content)"),
                {"id": file_id, "content": txt},
            )
        db.commit()
    except Exception:
        # If FTS is missing, we silently continue with metadata-only features.
        try:
            db.rollback()
        except Exception:
            pass


def _normalize_fts_query(q: str) -> str:
    # Very small normalization: trim and replace common separators with spaces.
    q = (q or "").strip()
    q = q.replace(",", " ").replace(";", " ").replace("\n", " ")
    q = " ".join(q.split())
    return q[:256]


def fts_search_ids(db: Session, q: str, limit: int = 20000) -> list[str]:
    """Return file_ids matching full-text query."""
    if not _fts_enabled():
        return []
    qn = _normalize_fts_query(q)
    if not qn:
        return []
    try:
        rows = db.execute(
            text(
                "SELECT file_id FROM file_content_fts "
                "WHERE file_content_fts MATCH :q "
                "LIMIT :lim"
            ),
            {"q": qn, "lim": int(limit)},
        ).all()
        return [str(r[0]) for r in rows if r and r[0]]
    except Exception:
        return []


def create_job(db: Session, folder: str, hash_mode: str) -> ScanJob:
    job = ScanJob(folder=folder, hash_mode=hash_mode, status="queued", created_at_utc=_now())
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> ScanJob | None:
    return db.get(ScanJob, job_id)


def list_jobs(db: Session, limit: int = 50) -> list[ScanJob]:
    stmt = select(ScanJob).order_by(ScanJob.created_at_utc.desc()).limit(limit)
    return list(db.scalars(stmt))


def set_job_status(db: Session, job_id: str, status: str, error_message: str | None = None) -> None:
    job = db.get(ScanJob, job_id)
    if not job:
        return
    job.status = status
    if status == "running":
        job.started_at_utc = _now()
        job.error_message = None
    if status in ("done", "error"):
        job.finished_at_utc = _now()
    if error_message:
        job.error_message = error_message
    db.add(job)
    db.commit()


def bump_job_counters(db: Session, job_id: str, **deltas: int) -> None:
    job = db.get(ScanJob, job_id)
    if not job:
        return
    for k, v in deltas.items():
        if hasattr(job, k) and isinstance(v, int):
            setattr(job, k, int(getattr(job, k) or 0) + v)
    db.add(job)
    db.commit()


def get_or_create_tag(db: Session, name: str) -> Tag:
    name = name.strip()
    tag = db.scalars(select(Tag).where(Tag.name == name)).first()
    if tag:
        return tag
    tag = Tag(name=name)
    db.add(tag)
    db.commit()
    db.refresh(tag)
    return tag


def set_file_tags(db: Session, rec: FileRecord, tag_names: list[str]) -> None:
    tags = [get_or_create_tag(db, n) for n in sorted(set([t.strip() for t in tag_names if t.strip()]))]
    rec.tags = tags
    rec.updated_at_utc = _now()
    db.add(rec)
    db.commit()


def upsert_file(
    db: Session,
    *,
    path: str,
    filename: str,
    extension: str,
    file_type: str,
    size_bytes: int,
    ctime_utc: datetime,
    mtime_utc: datetime,
    sha256: str | None,
    hash_mode: str,
    is_duplicate: bool,
    metadata: dict[str, Any] | None,
) -> tuple[FileRecord, str]:
    stmt = select(FileRecord).where(FileRecord.path == path)
    rec = db.scalars(stmt).first()
    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

    if rec is None:
        rec = FileRecord(
            path=path,
            filename=filename,
            extension=extension,
            file_type=file_type,
            size_bytes=size_bytes,
            ctime_utc=ctime_utc,
            mtime_utc=mtime_utc,
            sha256=sha256,
            hash_mode=hash_mode,
            is_duplicate=is_duplicate,
            metadata_json=meta_json,
            created_at_utc=_now(),
            updated_at_utc=_now(),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec, "inserted"

    changed = False
    for k, v in {
        "filename": filename,
        "extension": extension,
        "file_type": file_type,
        "size_bytes": size_bytes,
        "ctime_utc": ctime_utc,
        "mtime_utc": mtime_utc,
        "sha256": sha256,
        "hash_mode": hash_mode,
        "is_duplicate": is_duplicate,
        "metadata_json": meta_json,
    }.items():
        if getattr(rec, k) != v:
            setattr(rec, k, v)
            changed = True

    if changed:
        rec.updated_at_utc = _now()
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec, "updated"

    return rec, "unchanged"


def find_duplicate_by_hash(db: Session, sha256: str, hash_mode: str) -> bool:
    if not sha256:
        return False
    stmt = select(func.count()).select_from(FileRecord).where(
        and_(FileRecord.sha256 == sha256, FileRecord.hash_mode == hash_mode)
    )
    cnt = db.execute(stmt).scalar_one()
    return int(cnt) > 0


def search_files(
    db: Session,
    *,
    q: str | None = None,
    content_q: str | None = None,
    file_types: list[str] | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    sample_id: str | None = None,
    patient: str | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    modified_after: datetime | None = None,
    modified_before: datetime | None = None,
    duplicates_only: bool = False,
    sort: str = "mtime_desc",
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[FileRecord], int]:
    stmt = select(FileRecord).distinct()
    where = []

    if q:
        like = f"%{q.strip()}%"
        where.append(or_(
            FileRecord.filename.ilike(like),
            FileRecord.path.ilike(like),
            FileRecord.project.ilike(like),
            FileRecord.sample_id.ilike(like),
            FileRecord.patient_pseudo_id.ilike(like),
            FileRecord.metadata_json.ilike(like),
        ))

    # Full-text search inside file *content* (supported formats only, limited indexing).
    # Implemented via SQLite FTS5 virtual table.
    if content_q and _fts_enabled():
        ids = fts_search_ids(db, content_q)
        if not ids:
            return [], 0
        where.append(FileRecord.id.in_(ids))

    if file_types:
        where.append(FileRecord.file_type.in_(file_types))

    if project:
        where.append(FileRecord.project == project)

    if sample_id:
        where.append(FileRecord.sample_id == sample_id)

    if patient:
        where.append(FileRecord.patient_pseudo_id == patient)

    if min_size is not None:
        where.append(FileRecord.size_bytes >= int(min_size))
    if max_size is not None:
        where.append(FileRecord.size_bytes <= int(max_size))

    if modified_after:
        where.append(FileRecord.mtime_utc >= modified_after)
    if modified_before:
        where.append(FileRecord.mtime_utc <= modified_before)

    if duplicates_only:
        where.append(FileRecord.is_duplicate.is_(True))

    if tags:
        stmt = stmt.join(FileRecord.tags)
        where.append(Tag.name.in_(tags))

    if where:
        stmt = stmt.where(and_(*where))

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()

    sort = (sort or "mtime_desc").lower()
    if sort == "mtime_asc":
        stmt = stmt.order_by(FileRecord.mtime_utc.asc())
    elif sort == "size_asc":
        stmt = stmt.order_by(FileRecord.size_bytes.asc())
    elif sort == "size_desc":
        stmt = stmt.order_by(FileRecord.size_bytes.desc())
    else:
        stmt = stmt.order_by(FileRecord.mtime_utc.desc())

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    items = list(db.scalars(stmt))
    return items, int(total)
