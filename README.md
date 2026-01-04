# BioMed File Registry (MVP)

Лек, реално работещ инструмент за **регистър + индексатор** на големи биомедицински файлове.
Работи **само с метаданни** (не качва файловете и не чете съдържанието им целиком).

## Какво прави
- Импорт (сканиране) на папка рекурсивно
- Разпознаване на тип файл (FASTQ/FASTA/VCF/DICOM/CSV/TSV/JSON/XML/PDF/NIFTI/...)
- Извличане на **леки** метаданни според типа (header/columns/DICOM tags без пиксели)
- Индексиране в SQLite (локално)
- Търсене/филтри/сортиране през UI и REST API
- Експорт (CSV/JSON)
 - (Ново) По желание: **пълнотекстово търсене в съдържание** за текстови формати (SQLite FTS5)

## Стартиране
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Отвори:
- UI: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs

## Бележки
- База данни: `biomed_registry.db` (в проекта)
- Индексиране: от „Начало“ → „Индексирай папка“
- За DICOM използва `pydicom` с `stop_before_pixels=True`, за да не зарежда пиксели.
- Пълнотекстовото търсене индексира само **ограничена част** от файловете (първите N байта/реда),
   за да е бързо и безопасно за големи файлове. Контролира се чрез env:
   `BIOMED_FTS_ENABLED`, `BIOMED_FTS_MAX_BYTES`, `BIOMED_FTS_FASTQ_MAX_LINES`.
