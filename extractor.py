from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import fitz
import yake

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}
KEYWORD_SOURCE_LIMIT = 200_000


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_language(text: str) -> str:
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "ru" if cyrillic > latin else "en"


def _build_keyword_extractor(language: str, max_keywords: int) -> yake.KeywordExtractor:
    return yake.KeywordExtractor(
        lan=language,
        n=3,
        dedupLim=0.9,
        dedupFunc="seqm",
        windowsSize=1,
        top=max_keywords,
    )


def _read_text_file(file_path: Path) -> str:
    raw = file_path.read_bytes()
    for encoding in ("utf-8", "cp1251", "cp866"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_pdf_payload(file_name: str, file_bytes: bytes) -> tuple[str, list[dict[str, Any]]]:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover - depends on external parser
        raise ValueError(f"Не удалось открыть PDF-файл {file_name}") from exc

    page_summaries: list[dict[str, Any]] = []
    text_parts: list[str] = []

    try:
        for page_number, page in enumerate(document, start=1):
            page_text = _clean_text(page.get_text("text", sort=True))
            page_summaries.append(
                {
                    "page": page_number,
                    "chars": len(page_text),
                    "preview": page_text[:500],
                }
            )
            if page_text:
                text_parts.append(page_text)
    finally:
        document.close()

    return "\n\n".join(text_parts).strip(), page_summaries


def _extract_word_payload(file_name: str, file_bytes: bytes, extension: str) -> tuple[str, list[dict[str, Any]]]:
    soffice_path = shutil.which("soffice") or shutil.which("libreoffice")
    catdoc_path = shutil.which("catdoc")

    with tempfile.TemporaryDirectory(prefix="bot_extract_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / file_name
        input_path.write_bytes(file_bytes)
        output_path = temp_path / f"{input_path.stem}.txt"

        if soffice_path:
            conversion = subprocess.run(
                [
                    soffice_path,
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    temp_dir,
                    str(input_path),
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if conversion.returncode != 0 and not output_path.exists():
                raise ValueError(
                    f"LibreOffice не смог обработать файл {file_name}: "
                    f"{conversion.stderr.strip() or conversion.stdout.strip()}"
                )

        if not output_path.exists() and extension == ".doc" and catdoc_path:
            conversion = subprocess.run(
                [catdoc_path, str(input_path)],
                capture_output=True,
                timeout=60,
            )
            if conversion.returncode == 0:
                output_path.write_bytes(conversion.stdout)

        if not output_path.exists():
            raise ValueError(f"Не удалось извлечь текст из Word-файла {file_name}")

        text = _clean_text(_read_text_file(output_path))
        pages = [
            {
                "page": 1,
                "chars": len(text),
                "preview": text[:500],
            }
        ]
        return text, pages


def _extract_keywords(text: str, *, language: str, max_keywords: int) -> list[dict[str, Any]]:
    keyword_source = text[:KEYWORD_SOURCE_LIMIT]
    extractor = _build_keyword_extractor(language=language, max_keywords=max_keywords)
    return [
        {"keyword": keyword, "score": round(float(score), 4)}
        for keyword, score in extractor.extract_keywords(keyword_source)
    ]


def extract_document(
    file_name: str,
    file_bytes: bytes,
    *,
    language: str = "auto",
    max_keywords: int = 15,
) -> dict[str, Any]:
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Поддерживаются только файлы: {supported}")

    if extension == ".pdf":
        full_text, page_summaries = _extract_pdf_payload(file_name, file_bytes)
    else:
        full_text, page_summaries = _extract_word_payload(file_name, file_bytes, extension)

    if language == "auto":
        language = _guess_language(full_text)

    keywords = _extract_keywords(full_text, language=language, max_keywords=max_keywords)

    return {
        "file_name": file_name,
        "file_type": extension.lstrip("."),
        "file_size_bytes": len(file_bytes),
        "page_count": len(page_summaries),
        "char_count": len(full_text),
        "word_count": len(full_text.split()),
        "language": language,
        "keywords": keywords,
        "pages": page_summaries,
        "text": full_text,
    }


def export_results_json(results: list[dict[str, Any]]) -> str:
    return json.dumps({"documents": results}, ensure_ascii=False, indent=2)
