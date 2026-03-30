from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from extractor import extract_document

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "base"
TEST_DIR = PROJECT_DIR / "test"
DATA_DIR = PROJECT_DIR / "data"
BASE_LABELS_PATH = DATA_DIR / "base_document_labels.json"
BASE_CACHE_PATH = DATA_DIR / "base_document_cache.json"
TEST_CACHE_PATH = DATA_DIR / "test_document_cache.json"
RUNTIME_CORPUS_PATH = DATA_DIR / "runtime_training_corpus.json"
RUNTIME_DICTIONARY_PATH = DATA_DIR / "runtime_dictionary.json"
BASE_KEYWORD_LIMIT = 120
RUNTIME_DICTIONARY_LIMIT = 300

DEFAULT_LABELS = {
    "2022 Generation IV International Forum GIF Annual Report 2022.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Годовой отчет GIF по реакторам IV поколения и дорожным картам систем."
    },
    "Building-a-multiscale-framework-.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Теплогидравлический мультифизический контур для advanced reactors."
    },
    "DOE Releases Positive EIA(1).docx": {
        "label": "gen4",
        "include": False,
        "note": "Дубликат Natrium-новости, исключен из обучения."
    },
    "DOE Releases Positive EIA.docx": {
        "label": "gen4",
        "include": True,
        "note": "TerraPower Natrium и Generation IV контекст."
    },
    "Digital Twins for Nuclear Power Plants and Facilities.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Материал про digital twin для ядерных установок добавлен в тематический корпус advanced reactors."
    },
    "Elsevier A digital twin framework.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Цифровой двойник для Gen-IV FHR с явной привязкой к IV поколению."
    },
    "Elsevier Corrosion phenomena.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Коррозионные явления в супер-критической воде для реакторов Generation IV."
    },
    "Elsevier Enhancing multi-physics modeling.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Мультифизическое моделирование MSR и машинное обучение для next-generation reactors."
    },
    "Elsevier Gen 4 pool reactors.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Gen-IV pool reactors и валидируемая динамика теплоносителя на примере ALFRED."
    },
    "Elsevier Generation 4 International Forum.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Обзор десятилетнего прогресса Generation IV International Forum."
    },
    "Elsevier Roun-robin analysis.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Литиевые материалы для приложений Generation IV, связаны с MSR-тематикой."
    },
    "Elsevier Safety assessment of MSR.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Оценка безопасности molten salt reactor по методологии INPRO."
    },
    "Elsevier Stability and bifurcation analysis.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Устойчивость и бифуркации моделей реакторов IV поколения."
    },
    "GIF 2023 Annual Report Compressed.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Годовой отчет GIF 2023 по реакторным системам IV поколения."
    },
    "GIF Annual Report 2021.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Годовой отчет GIF 2021 с обзором семейств реакторов IV поколения."
    },
    "GIF Annual Report 2022 (1).pdf": {
        "label": "gen4",
        "include": False,
        "note": "Дубликат годового отчета GIF 2022, исключен из обучения."
    },
    "GIF_2024_Annual_Report.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Годовой отчет GIF 2024 по актуальному состоянию проектов Generation IV."
    },
    "High-Fidelity CFD Simulation of Mixed-Convection.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Kairos Power, pebble bed test reactor, advanced reactor R&D."
    },
    "IAEA FR17.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Материалы FR17 по fast reactors и related fuel cycles как части Gen IV-повестки."
    },
    "IAEA Next Gen Nuclear Reactors.docx": {
        "label": "gen4",
        "include": True,
        "note": "Публикация IAEA и GIF о ускоренном внедрении next generation reactors."
    },
    "IAEA coolants for FR.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Базовые жидкометаллические теплоносители для fast reactors."
    },
    "IAEA ИНПРО.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Методология ИНПРО используется в работах по advanced reactors и включена в эталонный корпус."
    },
    "IAEA_'KP-FHR Fuel Performance Methodology'.docx": {
        "label": "gen4",
        "include": True,
        "note": "KP-FHR, TRISO, fluoride salt-cooled high-temperature reactor."
    },
    "High-Fidelity CFD Simulation of Mixed-Convection.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Kairos Power, pebble bed test reactor, advanced reactor R&D."
    },
    "Reducing Proliferation Risks.pdf": {
        "label": "gen4",
        "include": True,
        "note": "TRISO, HALEU и advanced fuel context."
    },
    "TaF An Integral Metallic-Fuled and Lead-Cooled Reactor concept.pdf": {
        "label": "gen4",
        "include": True,
        "note": "Концепт lead-cooled fast reactor для 4th Generation Nuclear Energy Systems."
    },
    "WNN Kairos and DOE.docx": {
        "label": "gen4",
        "include": True,
        "note": "Короткая публикация WNN по Kairos и advanced reactor design."
    },
    "WNN_'Hermes 2 construction permits approved.docx": {
        "label": "gen4",
        "include": True,
        "note": "Kairos Hermes 2, molten salt-cooled reactors."
    },
    "WNN_'Kairos,_DOE_enhance_collaboration_on_advanced_reactor_design'.docx": {
        "label": "gen4",
        "include": True,
        "note": "Advanced reactor design, Kairos Power, ORNL."
    },
    "atomic energy МФП 20 лет.docx": {
        "label": "gen4",
        "include": True,
        "note": "Короткая новостная заметка о GIF и реакторах поколения IV."
    },
    "В_МИФИ_рассказали_о_перспективах_и_недостатках_проекта_малого_реактора.docx": {
        "label": "gen4",
        "include": True,
        "note": "Русскоязычная новость про Natrium и IV поколение."
    },
}

ALLOWED_LABELS = {"gen4", "other", "skip"}
DICTIONARY_ANCHORS = {
    "generation iv", "generation-iv", "generation 4", "4th generation", "gen iv",
    "gen-iv", "advanced reactor", "advanced reactors", "next generation reactor",
    "next-generation reactor", "fast reactor", "fast reactors", "molten salt",
    "molten-salt", "salt reactor", "salt reactors", "lead-cooled", "lead cooled",
    "sodium-cooled", "sodium cooled", "supercritical water", "very high temperature",
    "vhtr", "gfr", "lfr", "msr", "scwr", "sfr", "gif", "natrium", "terrapower",
    "kairos", "hermes", "triso", "haleu", "fhr", "gfhr", "kp-fhr", "fluoride",
    "ardp", "pebble", "liquid metal", "metallic-fueled", "metallic fueled",
    "замкнут", "быстр", "реактор iv", "поколен", "реактор iv поколения", "реакторы iv поколения",
    "натриев", "свинцов", "расплав", "соль", "солев", "быстрых реакторов", "реакторов поколения iv",
}
DICTIONARY_STOP_PATTERNS = (
    "doc number",
    "number rev",
    "effective date",
    "non proprietary",
    "david dalton",
    "bill gates",
    "created september",
    "modified october",
    "yahoo com",
    "nuclear reactors",
    "annual report",
    "working group",
    "united states",
    "technical secretariat",
    "policy group",
    "figure",
    "source",
    "journal homepage",
    "contents lists",
    "science direct",
    "china national",
)
ONE_WORD_WHITELIST = {
    "natrium", "kairos", "triso", "haleu", "hermes", "terrapower", "kp-fhr",
    "fhr", "gfhr", "ardp", "mcre", "mcfr", "msr", "lfr", "gfr", "sfr", "vhtr",
    "зятц", "брест", "bn-800", "bn800", "одэк",
}


def _guess_language_from_text(text: str) -> str:
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "ru" if cyrillic > latin else "en"


def list_base_files() -> list[Path]:
    if not BASE_DIR.exists():
        return []
    return sorted(path for path in BASE_DIR.iterdir() if path.is_file())


def list_test_files() -> list[Path]:
    if not TEST_DIR.exists():
        return []
    return sorted(path for path in TEST_DIR.rglob("*") if path.is_file())


def load_base_labels() -> list[dict[str, Any]]:
    existing = {}
    if BASE_LABELS_PATH.exists():
        existing = {
            item["file_name"]: item
            for item in json.loads(BASE_LABELS_PATH.read_text(encoding="utf-8")).get("documents", [])
        }

    rows: list[dict[str, Any]] = []
    for path in list_base_files():
        default = DEFAULT_LABELS.get(
            path.name,
            {
                "label": "gen4",
                "include": True,
                "note": "Новый документ эталонного корпуса Generation IV.",
            },
        )
        row = {
            "file_name": path.name,
            "path": str(path),
            "label": default["label"],
            "include": default["include"],
            "note": default["note"],
        }
        row.update(existing.get(path.name, {}))
        if row["label"] not in ALLOWED_LABELS:
            row["label"] = default["label"]
        rows.append(row)
    return rows


def save_base_labels(rows: list[dict[str, Any]]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [
            {
                "file_name": row["file_name"],
                "path": row["path"],
                "label": row["label"],
                "include": bool(row["include"]),
                "note": row.get("note", ""),
            }
            for row in rows
        ],
    }
    BASE_LABELS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_base_cache() -> dict[str, Any]:
    if not BASE_CACHE_PATH.exists():
        return {"documents": []}
    return json.loads(BASE_CACHE_PATH.read_text(encoding="utf-8"))


def build_base_cache(force: bool = False) -> dict[str, Any]:
    existing = {
        item["file_name"]: item
        for item in load_base_cache().get("documents", [])
    }
    documents: list[dict[str, Any]] = []

    for path in list_base_files():
        stat = path.stat()
        cached = existing.get(path.name)
        if (
            not force
            and cached
            and cached.get("size") == stat.st_size
            and cached.get("mtime") == stat.st_mtime
        ):
            documents.append(cached)
            continue

        raw = extract_document(path.name, path.read_bytes(), language="en", max_keywords=BASE_KEYWORD_LIMIT)
        language = _guess_language_from_text(raw["text"])
        if language != raw["language"]:
            raw = extract_document(
                path.name,
                path.read_bytes(),
                language=language,
                max_keywords=BASE_KEYWORD_LIMIT,
            )

        documents.append(
            {
                "file_name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "file_type": raw["file_type"],
                "language": raw["language"],
                "page_count": raw["page_count"],
                "word_count": raw["word_count"],
                "char_count": raw["char_count"],
                "preview": raw["text"][:1000],
                "text": raw["text"][:120000],
                "keywords": raw["keywords"][:BASE_KEYWORD_LIMIT],
            }
        )

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    BASE_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _infer_test_label(path: Path) -> str:
    path_str = str(path).lower()
    if "4 поколение" in path_str:
        return "gen4"
    return "other"


def build_test_cache(force: bool = False) -> dict[str, Any]:
    existing = {
        item["path"]: item
        for item in json.loads(TEST_CACHE_PATH.read_text(encoding="utf-8")).get("documents", [])
    } if TEST_CACHE_PATH.exists() else {}
    documents: list[dict[str, Any]] = []

    for path in list_test_files():
        stat = path.stat()
        cache_key = str(path)
        cached = existing.get(cache_key)
        if (
            not force
            and cached
            and cached.get("size") == stat.st_size
            and cached.get("mtime") == stat.st_mtime
        ):
            documents.append(cached)
            continue

        raw = extract_document(path.name, path.read_bytes(), language="en", max_keywords=BASE_KEYWORD_LIMIT)
        language = _guess_language_from_text(raw["text"])
        if language != raw["language"]:
            raw = extract_document(
                path.name,
                path.read_bytes(),
                language=language,
                max_keywords=BASE_KEYWORD_LIMIT,
            )

        documents.append(
            {
                "file_name": path.name,
                "path": str(path),
                "group": path.parent.name,
                "expected_label": _infer_test_label(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "file_type": raw["file_type"],
                "language": raw["language"],
                "page_count": raw["page_count"],
                "word_count": raw["word_count"],
                "char_count": raw["char_count"],
                "preview": raw["text"][:1000],
                "text": raw["text"][:120000],
                "keywords": raw["keywords"][:BASE_KEYWORD_LIMIT],
            }
        )

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    TEST_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _normalize_candidate(text: str) -> str:
    text = text.lower().replace("\ufeff", " ")
    text = re.sub(r"[^a-zа-яё0-9+\-/ ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _filename_to_phrase(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = stem.replace("_", " ").replace("-", " ").replace("'", " ")
    stem = re.sub(r"\(\d+\)", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def _is_meaningful_dictionary_candidate(normalized: str, doc_count: int) -> bool:
    if len(normalized) < 4 or len(normalized.split()) > 6:
        return False
    if any(pattern in normalized for pattern in DICTIONARY_STOP_PATTERNS):
        return False
    if len(normalized.split()) == 1:
        return normalized in ONE_WORD_WHITELIST
    return any(anchor in normalized for anchor in DICTIONARY_ANCHORS)


def _is_fallback_dictionary_candidate(normalized: str) -> bool:
    if len(normalized) < 4 or len(normalized.split()) > 6:
        return False
    if any(pattern in normalized for pattern in DICTIONARY_STOP_PATTERNS):
        return False
    if len(normalized.split()) == 1:
        return normalized in ONE_WORD_WHITELIST
    return True


def build_runtime_resources(force_cache: bool = False) -> dict[str, Any]:
    labels = load_base_labels()
    cache = {
        item["file_name"]: item
        for item in build_base_cache(force=force_cache).get("documents", [])
    }

    training_documents: list[dict[str, Any]] = []
    positive_docs: list[dict[str, Any]] = []

    for row in labels:
        if not row["include"] or row["label"] == "skip":
            continue
        cached = cache.get(row["file_name"])
        if not cached:
            continue
        document = {
            "id": row["file_name"],
            "label": row["label"],
            "file_name": row["file_name"],
            "text": cached["text"],
            "language": cached["language"],
            "keywords": cached["keywords"],
            "preview": cached["preview"],
            "note": row.get("note", ""),
        }
        training_documents.append(document)
        if row["label"] == "gen4":
            positive_docs.append(document)

    if not positive_docs:
        positive_docs = training_documents[:]

    runtime_corpus = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [
            {
                "id": doc["id"],
                "label": doc["label"],
                "text": doc["text"],
                "file_name": doc["file_name"],
                "language": doc["language"],
                "keywords": doc["keywords"],
                "note": doc["note"],
            }
            for doc in training_documents
        ],
    }
    RUNTIME_CORPUS_PATH.write_text(
        json.dumps(runtime_corpus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    candidate_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"docs": set(), "score_sum": 0.0, "count": 0, "surface": None}
    )
    for doc in positive_docs:
        title_candidate = _filename_to_phrase(doc["file_name"])
        for candidate in [title_candidate] + [item["keyword"] for item in doc["keywords"]]:
            normalized = _normalize_candidate(candidate)
            if len(normalized) < 4 or len(normalized.split()) > 6:
                continue
            stats = candidate_stats[normalized]
            stats["docs"].add(doc["file_name"])
            stats["count"] += 1
            if stats["surface"] is None:
                stats["surface"] = candidate.strip()
        for item in doc["keywords"]:
            normalized = _normalize_candidate(item["keyword"])
            if normalized in candidate_stats:
                candidate_stats[normalized]["score_sum"] += max(0.0001, float(item["score"]))

    ranked_terms = []
    for normalized, stats in candidate_stats.items():
        doc_count = len(stats["docs"])
        if doc_count == 0:
            continue
        if not _is_meaningful_dictionary_candidate(normalized, doc_count):
            continue
        avg_score = stats["score_sum"] / max(stats["count"], 1)
        rank = doc_count * 5 + (1 / max(avg_score, 0.0001))
        ranked_terms.append(
            {
                "canonical": stats["surface"] or normalized,
                "normalized": normalized,
                "doc_count": doc_count,
                "avg_score": round(avg_score, 4),
                "rank": rank,
            }
        )

    ranked_terms.sort(key=lambda item: item["rank"], reverse=True)

    selected_terms = ranked_terms[:RUNTIME_DICTIONARY_LIMIT]
    if len(selected_terms) < RUNTIME_DICTIONARY_LIMIT:
        selected_norms = {item["normalized"] for item in selected_terms}
        fallback_terms = []
        for normalized, stats in candidate_stats.items():
            if normalized in selected_norms:
                continue
            if not _is_fallback_dictionary_candidate(normalized):
                continue
            avg_score = stats["score_sum"] / max(stats["count"], 1)
            fallback_terms.append(
                {
                    "canonical": stats["surface"] or normalized,
                    "normalized": normalized,
                    "doc_count": len(stats["docs"]),
                    "avg_score": round(avg_score, 4),
                    "rank": len(stats["docs"]) * 3 + (1 / max(avg_score, 0.0001)),
                }
            )
        fallback_terms.sort(key=lambda item: item["rank"], reverse=True)
        for item in fallback_terms:
            if len(selected_terms) >= RUNTIME_DICTIONARY_LIMIT:
                break
            selected_terms.append(item)
            selected_norms.add(item["normalized"])

    dictionary_entries = []
    for item in selected_terms:
        dictionary_entries.append(
            {
                "canonical": item["canonical"],
                "category": "learned_from_base",
                "weight": round(min(3.0, 1.0 + item["doc_count"] * 0.35), 2),
                "variants": [item["canonical"], item["normalized"]],
                "doc_count": item["doc_count"],
                "avg_score": item["avg_score"],
            }
        )

    runtime_dictionary = {
        "name": "Словарь, собранный по документам из base",
        "topic": "generation_iv_reactors_base",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": dictionary_entries,
    }
    RUNTIME_DICTIONARY_PATH.write_text(
        json.dumps(runtime_dictionary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "training_documents": len(training_documents),
        "positive_documents": len(positive_docs),
        "dictionary_entries": len(dictionary_entries),
    }


def get_base_overview() -> list[dict[str, Any]]:
    labels = {
        row["file_name"]: row
        for row in load_base_labels()
    }
    cache = {
        item["file_name"]: item
        for item in build_base_cache(force=False).get("documents", [])
    }
    rows = []
    for file_name, label_info in labels.items():
        cached = cache.get(file_name, {})
        rows.append(
            {
                "include": bool(label_info["include"]),
                "label": label_info["label"],
                "file_name": file_name,
                "file_type": cached.get("file_type", ""),
                "language": cached.get("language", ""),
                "pages": cached.get("page_count", 0),
                "words": cached.get("word_count", 0),
                "note": label_info.get("note", ""),
                "preview": cached.get("preview", "")[:240],
            }
        )
    return rows


def get_test_overview() -> list[dict[str, Any]]:
    rows = []
    for item in build_test_cache(force=False).get("documents", []):
        rows.append(
            {
                "group": item.get("group", ""),
                "expected_label": item.get("expected_label", ""),
                "file_name": item.get("file_name", ""),
                "file_type": item.get("file_type", ""),
                "language": item.get("language", ""),
                "pages": item.get("page_count", 0),
                "words": item.get("word_count", 0),
                "preview": item.get("preview", "")[:240],
            }
        )
    return rows
