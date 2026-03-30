from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from theme_dictionary import load_seed_dictionary
from training_manager import build_test_cache

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "gen4_training_corpus.json"
RUNTIME_CORPUS_PATH = DATA_DIR / "runtime_training_corpus.json"
MAX_FEATURES = 400


def _normalize_text(text: str) -> str:
    text = text.lower().replace("\ufeff", " ")
    cleaned = []
    for char in text:
        if char.isalnum() or char in "+-/ ":
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return f" {' '.join(''.join(cleaned).split())} "


def _normalize_vector(values: Counter[str]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in values.values()))
    if norm == 0:
        return {}
    return {token: value / norm for token, value in values.items()}


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass
class PredictionResult:
    label: str
    probability_gen4: float
    probability_other: float
    similarity: float
    threshold: float
    evidence_tokens: list[dict[str, Any]]


class DomainSimilarityThemeModel:
    def __init__(self, max_features: int = MAX_FEATURES) -> None:
        self.max_features = max_features
        self.reference_entries: list[dict[str, Any]] = []
        self.centroid: dict[str, float] = {}
        self.reference_similarities: list[float] = []
        self.reference_mean = 0.0
        self.reference_std = 0.0
        self.reference_min = 0.0
        self.reference_max = 0.0
        self.threshold = 0.2

    def _vectorize_text(self, text: str) -> dict[str, float]:
        normalized_text = _normalize_text(text)
        weighted = Counter()
        for entry in self.reference_entries:
            hit_count = 0
            for variant in entry["variants"]:
                normalized_variant = _normalize_text(variant).strip()
                if not normalized_variant:
                    continue
                hit_count = max(hit_count, normalized_text.count(f" {normalized_variant} "))
            if hit_count:
                weighted[entry["canonical"]] = float(hit_count) * float(entry.get("weight", 1.0))
        return _normalize_vector(weighted)

    def fit(self, texts: list[str]) -> "DomainSimilarityThemeModel":
        dictionary = load_seed_dictionary(target="model")
        self.reference_entries = dictionary.get("entries", [])[: self.max_features]
        document_vectors = [self._vectorize_text(text) for text in texts]
        centroid_counter = Counter()
        for vector in document_vectors:
            centroid_counter.update(vector)
        if document_vectors:
            centroid_counter = Counter(
                {token: value / len(document_vectors) for token, value in centroid_counter.items()}
            )
        self.centroid = _normalize_vector(centroid_counter)

        self.reference_similarities = [
            _cosine_similarity(vector, self.centroid)
            for vector in document_vectors
            if vector
        ]
        if self.reference_similarities:
            self.reference_mean = sum(self.reference_similarities) / len(self.reference_similarities)
            variance = sum(
                (value - self.reference_mean) ** 2 for value in self.reference_similarities
            ) / len(self.reference_similarities)
            self.reference_std = math.sqrt(variance)
            self.reference_min = min(self.reference_similarities)
            self.reference_max = max(self.reference_similarities)
            self.threshold = max(
                0.12,
                min(self.reference_mean - 1.5 * self.reference_std, self.reference_min),
                self.reference_min * 0.92,
            )
        return self

    def predict(self, text: str, threshold_override: float | None = None) -> PredictionResult:
        active_threshold = self.threshold if threshold_override is None else float(threshold_override)
        vector = self._vectorize_text(text)
        if not vector:
            return PredictionResult(
                label="other",
                probability_gen4=0.01,
                probability_other=0.99,
                similarity=0.0,
                threshold=round(active_threshold, 4),
                evidence_tokens=[],
            )
        similarity = _cosine_similarity(vector, self.centroid)
        scale = max(self.reference_std * 1.5, 0.035)
        probability_gen4 = round(_sigmoid((similarity - active_threshold) / scale), 4)
        probability_other = round(1.0 - probability_gen4, 4)
        label = "gen4" if similarity >= active_threshold else "other"

        evidence_rows: list[dict[str, Any]] = []
        for token, value in sorted(vector.items(), key=lambda item: item[1] * self.centroid.get(item[0], 0.0), reverse=True)[:20]:
            contribution = value * self.centroid.get(token, 0.0)
            if contribution <= 0:
                continue
            evidence_rows.append(
                {
                    "token": token,
                    "count": 1,
                    "direction": "gen4",
                    "weight": round(contribution, 4),
                }
            )

        return PredictionResult(
            label=label,
            probability_gen4=probability_gen4,
            probability_other=probability_other,
            similarity=round(similarity, 4),
            threshold=round(active_threshold, 4),
            evidence_tokens=evidence_rows[:10],
        )


@lru_cache(maxsize=1)
def load_training_corpus() -> list[dict[str, Any]]:
    active_path = RUNTIME_CORPUS_PATH if RUNTIME_CORPUS_PATH.exists() else CORPUS_PATH
    return json.loads(active_path.read_text(encoding="utf-8"))["documents"]


@lru_cache(maxsize=1)
def load_trained_theme_model() -> DomainSimilarityThemeModel:
    corpus = load_training_corpus()
    positive_texts = [item["text"] for item in corpus if item.get("label") == "gen4"] or [item["text"] for item in corpus]
    model = DomainSimilarityThemeModel(max_features=MAX_FEATURES)
    model.fit(positive_texts)
    return model


def analyze_text_theme(text: str, *, threshold: float | None = None) -> dict[str, Any]:
    prediction = load_trained_theme_model().predict(text, threshold_override=threshold)
    return {
        "prediction": {
            "label": prediction.label,
            "probability_gen4": prediction.probability_gen4,
            "probability_other": prediction.probability_other,
            "similarity": prediction.similarity,
            "threshold": prediction.threshold,
        },
        "evidence_tokens": prediction.evidence_tokens,
    }


def evaluate_theme_model() -> dict[str, Any]:
    test_documents = build_test_cache(force=False).get("documents", [])
    if not test_documents:
        model = load_trained_theme_model()
        positives = [item for item in load_training_corpus() if item.get("label") == "gen4"] or load_training_corpus()
        covered = 0
        for item in positives:
            if model.predict(item["text"]).label == "gen4":
                covered += 1
        total = len(positives) or 1
        return {
            "documents": len(positives),
            "accuracy_percent": round(covered / total * 100, 2),
            "confusion": {"gen4->gen4": covered, "gen4->other": len(positives) - covered, "other->gen4": 0, "other->other": 0},
        }

    model = load_trained_theme_model()
    correct = 0
    confusion = {"gen4->gen4": 0, "gen4->other": 0, "other->gen4": 0, "other->other": 0}
    for item in test_documents:
        predicted = model.predict(item["text"]).label
        expected = item["expected_label"]
        confusion[f"{expected}->{predicted}"] += 1
        if predicted == expected:
            correct += 1

    total = len(test_documents) or 1
    return {
        "documents": len(test_documents),
        "accuracy_percent": round(correct / total * 100, 2),
        "confusion": confusion,
    }


def reset_model_cache() -> None:
    load_training_corpus.cache_clear()
    load_trained_theme_model.cache_clear()
