from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
DICTIONARY_PATH = DATA_DIR / "gen4_seed_dictionary.json"
RUNTIME_DICTIONARY_PATH = DATA_DIR / "runtime_dictionary.json"
MANUAL_OVERRIDES_PATH = DATA_DIR / "manual_dictionary_overrides.json"
DEFAULT_YAKE_THRESHOLD = 0.24


def _normalize_text(text: str) -> str:
    text = text.lower().replace("\ufeff", " ")
    text = re.sub(r"[^a-zа-яё0-9+\-/ ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return f" {text.strip()} "


def _normalize_term_key(value: str) -> str:
    return _normalize_text(value).strip()


def _default_overrides_payload() -> dict[str, Any]:
    return {
        "updated_at": None,
        "targets": {
            "model": {"blocked_terms": [], "manual_entries": []},
            "yake": {"blocked_terms": [], "manual_entries": []},
        },
    }


@lru_cache(maxsize=1)
def load_manual_dictionary_overrides() -> dict[str, Any]:
    if not MANUAL_OVERRIDES_PATH.exists():
        return _default_overrides_payload()
    payload = json.loads(MANUAL_OVERRIDES_PATH.read_text(encoding="utf-8"))
    defaults = _default_overrides_payload()
    payload.setdefault("targets", {})
    for target, target_defaults in defaults["targets"].items():
        payload["targets"].setdefault(target, target_defaults)
        payload["targets"][target].setdefault("blocked_terms", [])
        payload["targets"][target].setdefault("manual_entries", [])
    return payload


def save_manual_dictionary_overrides(payload: dict[str, Any]) -> None:
    MANUAL_OVERRIDES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reset_dictionary_cache()


@lru_cache(maxsize=4)
def load_seed_dictionary(target: str = "yake") -> dict[str, Any]:
    active_path = RUNTIME_DICTIONARY_PATH if RUNTIME_DICTIONARY_PATH.exists() else DICTIONARY_PATH
    base_dictionary = json.loads(active_path.read_text(encoding="utf-8"))
    overrides = load_manual_dictionary_overrides()
    target_overrides = overrides["targets"].get(target, {"blocked_terms": [], "manual_entries": []})
    blocked_terms = {_normalize_term_key(term) for term in target_overrides.get("blocked_terms", [])}
    manual_entries = target_overrides.get("manual_entries", [])

    merged_entries: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for entry in base_dictionary.get("entries", []):
        canonical_key = _normalize_term_key(entry["canonical"])
        if canonical_key in blocked_terms:
            continue
        merged_entries.append(entry)
        seen_keys.add(canonical_key)

    for entry in manual_entries:
        canonical_key = _normalize_term_key(entry["canonical"])
        if not canonical_key or canonical_key in blocked_terms or canonical_key in seen_keys:
            continue
        merged_entries.append(
            {
                "canonical": entry["canonical"],
                "category": entry.get("category", f"manual_{target}"),
                "weight": float(entry.get("weight", 2.0)),
                "variants": entry.get("variants", [entry["canonical"], canonical_key]),
                "doc_count": entry.get("doc_count", 0),
                "avg_score": entry.get("avg_score", 0.0),
                "source": "manual",
            }
        )
        seen_keys.add(canonical_key)

    merged_dictionary = dict(base_dictionary)
    merged_dictionary["entries"] = merged_entries
    merged_dictionary["name"] = f"{base_dictionary.get('name', 'Словарь')} ({target})"
    return merged_dictionary


def reset_dictionary_cache() -> None:
    load_seed_dictionary.cache_clear()
    load_manual_dictionary_overrides.cache_clear()


def analyze_against_dictionary(text: str, keywords: list[dict[str, Any]]) -> dict[str, Any]:
    dictionary = load_seed_dictionary(target="yake")
    normalized_text = _normalize_text(text)

    matched_entries: list[dict[str, Any]] = []
    category_counter: Counter[str] = Counter()
    matched_variants: list[str] = []

    for entry in dictionary["entries"]:
        for variant in entry["variants"]:
            normalized_variant = _normalize_text(variant)
            if normalized_variant.strip() and normalized_variant in normalized_text:
                matched_entries.append(
                    {
                        "canonical": entry["canonical"],
                        "category": entry["category"],
                        "weight": entry["weight"],
                        "matched_variant": variant,
                    }
                )
                category_counter[entry["category"]] += 1
                matched_variants.append(variant)
                break

    total_weight = sum(item["weight"] for item in dictionary["entries"])
    matched_weight = sum(item["weight"] for item in matched_entries)
    coverage = round((matched_weight / total_weight) * 100, 2) if total_weight else 0.0

    keyword_hits = []
    for keyword in keywords:
        normalized_keyword = _normalize_text(keyword["keyword"])
        for entry in dictionary["entries"]:
            if any(_normalize_text(variant) in normalized_keyword for variant in entry["variants"]):
                keyword_hits.append(
                    {
                        "keyword": keyword["keyword"],
                        "score": keyword["score"],
                        "category": entry["category"],
                        "canonical": entry["canonical"],
                    }
                )
                break

    suggestions = build_dictionary_suggestions(keywords, matched_entries)

    return {
        "dictionary_name": dictionary["name"],
        "matched_entries": matched_entries,
        "matched_terms": sorted({item["canonical"] for item in matched_entries}),
        "matched_variants": sorted(set(matched_variants)),
        "matched_categories": dict(category_counter),
        "coverage_percent": coverage,
        "keyword_hits": keyword_hits,
        "suggestions": suggestions,
    }


def analyze_yake_theme(
    text: str,
    keywords: list[dict[str, Any]],
    *,
    dictionary_analysis: dict[str, Any] | None = None,
    threshold: float = DEFAULT_YAKE_THRESHOLD,
) -> dict[str, Any]:
    analysis = dictionary_analysis or analyze_against_dictionary(text, keywords)
    matched_terms = analysis.get("matched_terms", [])
    keyword_hits = analysis.get("keyword_hits", [])
    coverage = float(analysis.get("coverage_percent", 0.0))

    hit_strength = 0.0
    hit_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in keyword_hits:
        canonical = hit["canonical"]
        if canonical in seen:
            continue
        seen.add(canonical)
        strength = 1.0 / (1.0 + float(hit["score"]) * 8.0)
        hit_strength += strength
        hit_rows.append(
            {
                "canonical": canonical,
                "keyword": hit["keyword"],
                "score": hit["score"],
                "strength": round(strength, 4),
            }
        )

    matched_term_ratio = min(1.0, len(matched_terms) / 10.0)
    keyword_signal = min(1.0, hit_strength / 5.0)
    coverage_signal = min(1.0, coverage / 18.0)
    confidence = round(
        min(0.99, 0.25 * matched_term_ratio + 0.35 * keyword_signal + 0.40 * coverage_signal),
        4,
    )
    label = "gen4" if confidence >= threshold else "other"

    return {
        "prediction": {
            "label": label,
            "probability_gen4": confidence,
            "probability_other": round(1.0 - confidence, 4),
            "threshold": round(threshold, 4),
        },
        "matched_term_count": len(matched_terms),
        "matched_keyword_count": len(hit_rows),
        "coverage_percent": coverage,
        "keyword_signal": round(hit_strength, 4),
        "evidence_hits": sorted(hit_rows, key=lambda item: item["strength"], reverse=True)[:12],
    }


def build_dictionary_suggestions(
    keywords: list[dict[str, Any]], matched_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    known_terms = {item["canonical"] for item in matched_entries}
    suggestions: list[dict[str, Any]] = []
    seen: set[str] = set()

    for keyword in keywords:
        candidate = keyword["keyword"].strip()
        normalized = _normalize_text(candidate).strip()
        if (
            len(candidate) < 5
            or candidate.lower() in seen
            or candidate.lower() in {term.lower() for term in known_terms}
            or normalized.isdigit()
        ):
            continue
        suggestions.append({"candidate": candidate, "yake_score": keyword["score"]})
        seen.add(candidate.lower())
        if len(suggestions) == 10:
            break

    return suggestions


def aggregate_dictionary_growth(results: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        item
        for item in results
        if item.get("theme_analysis", {}).get("prediction", {}).get("label") == "gen4"
    ]

    aggregated: Counter[str] = Counter()
    for item in accepted:
        for suggestion in item.get("dictionary_analysis", {}).get("suggestions", []):
            aggregated[suggestion["candidate"]] += 1

    candidates = [
        {"candidate": candidate, "documents": count}
        for candidate, count in aggregated.most_common(20)
    ]

    return {
        "accepted_documents": len(accepted),
        "candidate_terms": candidates,
    }
