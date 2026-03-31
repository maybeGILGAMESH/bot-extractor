from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from extractor import export_results_json, extract_document
from theme_dictionary import (
    DEFAULT_YAKE_THRESHOLD,
    aggregate_dictionary_growth,
    analyze_against_dictionary,
    analyze_yake_theme,
    load_manual_dictionary_overrides,
    load_seed_dictionary,
    reset_dictionary_cache,
    save_manual_dictionary_overrides,
)
from theme_model import (
    analyze_text_theme,
    evaluate_theme_model,
    load_training_corpus,
    load_trained_theme_model,
    reset_model_cache,
)
from training_manager import (
    build_base_cache,
    build_runtime_resources,
    get_base_overview,
    get_test_overview,
    load_base_labels,
    save_base_labels,
)

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_REGISTRY_PATH = PROJECT_DIR / "data" / "gen4_source_registry.json"

st.set_page_config(
    page_title="Бот извлекатель",
    page_icon="B",
    layout="wide",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Unbounded:wght@600;700;800&display=swap');

:root {
  --bg: #09121c;
  --bg-soft: rgba(14, 27, 40, 0.78);
  --card: rgba(246, 242, 232, 0.08);
  --line: rgba(255, 255, 255, 0.12);
  --text: #f7f1e6;
  --muted: #b8c4cf;
  --accent: #f2b84b;
  --accent-2: #f07c4a;
  --good: #93d6b5;
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(242, 184, 75, 0.18), transparent 30%),
    radial-gradient(circle at 85% 10%, rgba(240, 124, 74, 0.18), transparent 28%),
    linear-gradient(135deg, #071019 0%, #0d1b29 45%, #13283d 100%);
  color: var(--text);
}

html, body, [class*="css"]  {
  font-family: 'Manrope', sans-serif;
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stToolbar"] {
  right: 1rem;
}

[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed rgba(242, 184, 75, 0.55);
  background: rgba(8, 17, 27, 0.68);
  border-radius: 24px;
}

[data-testid="stForm"] {
  background: rgba(8, 17, 27, 0.7);
  border: 1px solid var(--line);
  border-radius: 28px;
  padding: 1rem 1rem 0.5rem 1rem;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
}

.stButton > button, .stDownloadButton > button, [data-testid="baseButton-secondary"] {
  background: linear-gradient(135deg, var(--accent) 0%, #f9d978 100%);
  color: #231706;
  border: 0;
  border-radius: 999px;
  font-weight: 800;
}

.hero {
  padding: 2rem 2rem 1.5rem 2rem;
  border-radius: 32px;
  background:
    linear-gradient(135deg, rgba(246, 242, 232, 0.08), rgba(246, 242, 232, 0.03)),
    rgba(8, 17, 27, 0.55);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(10px);
}

.eyebrow {
  display: inline-block;
  padding: 0.45rem 0.8rem;
  border-radius: 999px;
  background: rgba(147, 214, 181, 0.14);
  color: var(--good);
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-size: 0.78rem;
}

.hero h1 {
  margin: 1rem 0 0.5rem 0;
  font-family: 'Unbounded', sans-serif;
  font-size: clamp(1.6rem, 3.5vw, 3.3rem);
  line-height: 0.95;
  color: var(--text);
}

.hero h2 {
  margin: 0;
  font-size: clamp(1.4rem, 2vw, 2rem);
  font-weight: 700;
  color: #ffd896;
}

.hero p {
  max-width: 52rem;
  font-size: 1.05rem;
  color: var(--muted);
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1.25rem;
}

.feature-card {
  border-radius: 24px;
  padding: 1rem 1.1rem;
  background: var(--card);
  border: 1px solid var(--line);
}

.feature-card strong {
  display: block;
  margin-bottom: 0.45rem;
  color: var(--text);
}

.feature-card span {
  color: var(--muted);
  font-size: 0.96rem;
}

.feature-empty {
  border-radius: 24px;
  min-height: 100%;
}

.result-card {
  margin-top: 1rem;
  padding: 1rem 1.1rem;
  background: rgba(8, 17, 27, 0.72);
  border: 1px solid var(--line);
  border-radius: 22px;
}

.calc-note {
  padding: 0.9rem 1rem;
  border-radius: 18px;
  background: rgba(242, 184, 75, 0.1);
  border: 1px solid rgba(242, 184, 75, 0.2);
  color: var(--text);
}

.theme-card {
  margin-top: 1rem;
  padding: 1rem 1.1rem;
  border-radius: 22px;
  background: rgba(147, 214, 181, 0.08);
  border: 1px solid rgba(147, 214, 181, 0.18);
}

.decision-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.9rem;
  margin: 1rem 0 1rem 0;
}

.decision-card {
  border-radius: 20px;
  padding: 1rem 1.1rem;
  border: 1px solid rgba(255, 255, 255, 0.12);
}

.decision-card strong {
  display: block;
  margin-bottom: 0.35rem;
  font-size: 0.95rem;
}

.decision-card span {
  display: block;
  font-size: 1.05rem;
  font-weight: 800;
}

.decision-good {
  background: rgba(76, 175, 80, 0.16);
  border-color: rgba(76, 175, 80, 0.35);
  color: #dff6dd;
}

.decision-bad {
  background: rgba(229, 83, 75, 0.16);
  border-color: rgba(229, 83, 75, 0.38);
  color: #ffd9d6;
}

.tiny-note {
  color: var(--muted);
  font-size: 0.9rem;
}

.visual-shell {
  margin-top: 1.25rem;
  padding: 1.2rem 1.2rem 1.35rem 1.2rem;
  border-radius: 28px;
  background:
    radial-gradient(circle at 0% 0%, rgba(242, 184, 75, 0.18), transparent 30%),
    radial-gradient(circle at 100% 0%, rgba(147, 214, 181, 0.16), transparent 26%),
    rgba(8, 17, 27, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 22px 52px rgba(0, 0, 0, 0.24);
}

.visual-kicker {
  display: inline-block;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  background: rgba(242, 184, 75, 0.12);
  color: #ffe4a7;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.visual-shell h3 {
  margin: 0.9rem 0 0.4rem 0;
  font-family: 'Unbounded', sans-serif;
  font-size: clamp(1.4rem, 2.4vw, 2.2rem);
  line-height: 1.05;
}

.visual-shell p {
  margin: 0;
  color: var(--muted);
}

.visual-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.9rem;
  margin-top: 1rem;
}

.visual-card {
  position: relative;
  overflow: hidden;
  border-radius: 22px;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.visual-card::after {
  content: "";
  position: absolute;
  inset: auto -20% -55% auto;
  width: 110px;
  height: 110px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(242, 184, 75, 0.18), transparent 70%);
}

.visual-card strong {
  display: block;
  color: var(--muted);
  font-size: 0.86rem;
  font-weight: 700;
}

.visual-number {
  display: block;
  margin-top: 0.35rem;
  font-family: 'Unbounded', sans-serif;
  font-size: clamp(1.2rem, 2vw, 2rem);
  color: var(--text);
}

.visual-caption {
  display: block;
  margin-top: 0.25rem;
  color: #d6deea;
  font-size: 0.84rem;
}

.visual-pipeline {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.7rem;
  margin-top: 1rem;
  align-items: stretch;
}

.pipeline-step {
  position: relative;
  border-radius: 20px;
  padding: 0.95rem 0.9rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
  min-height: 96px;
}

.pipeline-step strong {
  display: block;
  font-size: 0.9rem;
  color: #fff1cf;
}

.pipeline-step span {
  display: block;
  margin-top: 0.35rem;
  color: var(--muted);
  font-size: 0.86rem;
}

.pipeline-step::before {
  content: "";
  position: absolute;
  top: 14px;
  right: 14px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #93d6b5;
  box-shadow: 0 0 0 rgba(147, 214, 181, 0.45);
  animation: pulseGlow 2.1s ease-in-out infinite;
}

@keyframes pulseGlow {
  0% {
    box-shadow: 0 0 0 0 rgba(147, 214, 181, 0.35);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(147, 214, 181, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(147, 214, 181, 0);
  }
}

@media (max-width: 900px) {
  .feature-grid {
    grid-template-columns: 1fr;
  }
  .visual-grid {
    grid-template-columns: 1fr 1fr;
  }
  .visual-pipeline {
    grid-template-columns: 1fr;
  }
}
</style>
"""


def render_hero() -> None:
    st.markdown(
        CUSTOM_CSS
        + """
<section class="hero">
  <span class="eyebrow">PyMuPDF + YAKE</span>
  <h1>Прототип тематического анализа документов</h1>
  <h2>Бот извлекатель</h2>
  <p>
    Сервис извлекает текст из PDF и Word-документов, выделяет ключевые фразы,
    сравнивает документ со словарем домена и рассчитывает похожесть на корпус
    материалов по реакторам IV поколения.
  </p>
  <div class="feature-grid">
    <div class="feature-card">
      <strong>Загрузка документов</strong>
      <span>Поддерживаются PDF, DOC и DOCX. Файлы обрабатываются в текущей сессии и сразу попадают в аналитический контур.</span>
    </div>
    <div class="feature-card">
      <strong>Извлечение текста</strong>
      <span>PDF разбираются через PyMuPDF, Word-файлы через LibreOffice, после чего YAKE извлекает ключевые фразы.</span>
    </div>
    <div class="feature-empty" aria-hidden="true"></div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, int, float, float]:
    model_eval = evaluate_theme_model()
    corpus = load_training_corpus()
    source_registry = json.loads(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))["sources"]
    active_dictionary = load_seed_dictionary()
    default_model_threshold = load_trained_theme_model().threshold
    with st.sidebar:
        st.markdown("## Параметры")
        language = st.selectbox(
            "Язык документа",
            options=["auto", "ru", "en"],
            index=0,
            help="Можно выбрать auto, чтобы сервис сам определял ru или en.",
        )
        max_keywords = st.slider(
            "Сколько ключевых фраз показать",
            min_value=20,
            max_value=400,
            value=80,
            step=10,
        )
        model_threshold = st.slider(
            "Порог нашей модели",
            min_value=0.05,
            max_value=0.95,
            value=float(round(default_model_threshold, 2)),
            step=0.01,
            help="Если similarity выше этого порога, модель считает документ относящимся к IV поколению.",
        )
        yake_threshold = st.slider(
            "Порог YAKE-решения",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_YAKE_THRESHOLD),
            step=0.01,
            help="Если итоговый confidence YAKE выше этого порога, YAKE считает документ относящимся к IV поколению.",
        )
        st.markdown("## Дипломный контур")
        st.write(
            "После извлечения сервис дает два независимых решения: "
            "по словарю и YAKE, а также по корпусной модели похожести на базу Generation IV."
        )
        st.metric("Эталонный корпус base", len(corpus))
        st.metric("Проверка на test", f"{model_eval['accuracy_percent']}%")
        st.metric("Активный словарь", len(active_dictionary.get("entries", [])))
        st.markdown(
            '<p class="tiny-note">Методика: <code>docs/gen4_methodology.md</code></p>',
            unsafe_allow_html=True,
        )
        with st.expander("Источники для диплома"):
            for source in source_registry:
                st.markdown(
                    f"- **{source['organization']}**: "
                    f"[{source['title']}]({source['url']})"
                )
        with st.expander("Тестовые файлы"):
            st.write(
                "Готовые документы для загрузки лежат в каталоге `sample_docs`."
            )
        with st.expander("Реальный корпус base"):
            st.write("Документы для обучения сейчас читаются из каталога `base`.")
        with st.expander("Отобранные термины словаря"):
            dictionary_rows = pd.DataFrame(active_dictionary.get("entries", [])).copy()
            if dictionary_rows.empty:
                st.info("Словарь пока не собран.")
            else:
                display = dictionary_rows.rename(
                    columns={
                        "canonical": "Термин",
                        "doc_count": "Документов",
                        "avg_score": "Средний YAKE score",
                    }
                )
                keep_columns = [col for col in ["Термин", "Документов", "Средний YAKE score"] if col in display.columns]
                st.dataframe(display[keep_columns], use_container_width=True, hide_index=True, height=320)
                with st.expander("Показать список терминов"):
                    for idx, term in enumerate(display["Термин"].tolist(), start=1):
                        st.markdown(f"{idx}. `{term}`")
        st.markdown('<p class="tiny-note">Файлы обрабатываются в памяти текущей сессии.</p>', unsafe_allow_html=True)
    return language, max_keywords, model_threshold, yake_threshold


def apply_theme_analyses(
    results: list[dict],
    *,
    model_threshold: float,
    yake_threshold: float,
) -> list[dict]:
    for result in results:
        result["dictionary_analysis"] = analyze_against_dictionary(
            result["text"],
            result["keywords"],
        )
        result["yake_theme_analysis"] = analyze_yake_theme(
            result["text"],
            result["keywords"],
            dictionary_analysis=result["dictionary_analysis"],
            threshold=yake_threshold,
        )
        result["theme_analysis"] = analyze_text_theme(
            result["text"],
            threshold=model_threshold,
        )
    return results


def process_files(
    uploads: list[st.runtime.uploaded_file_manager.UploadedFile],
    *,
    language: str,
    max_keywords: int,
    model_threshold: float,
    yake_threshold: float,
) -> tuple[list[dict], list[str]]:
    results: list[dict] = []
    errors: list[str] = []

    for uploaded_file in uploads:
        try:
            result = extract_document(
                uploaded_file.name,
                uploaded_file.getvalue(),
                language=language,
                max_keywords=max_keywords,
            )
            results.append(result)
        except ValueError as exc:
            errors.append(str(exc))

    return apply_theme_analyses(
        results,
        model_threshold=model_threshold,
        yake_threshold=yake_threshold,
    ), errors


def render_results(results: list[dict]) -> None:
    if not results:
        return

    total_pages = sum(item["page_count"] for item in results)
    total_words = sum(item["word_count"] for item in results)

    st.markdown("## Результаты")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Документов", len(results))
    metric_cols[1].metric("Страниц", total_pages)
    metric_cols[2].metric("Слов", total_words)
    metric_cols[3].metric("Собрано", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    export_payload = export_results_json(results)
    st.download_button(
        label="Скачать общий JSON",
        data=export_payload,
        file_name="bot_extract_results.json",
        mime="application/json",
        use_container_width=False,
    )

    for result in results:
        st.markdown(
            f'<div class="result-card"><strong>{result["file_name"]}</strong></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        cols[0].metric("Страниц", result["page_count"])
        cols[1].metric("Символов", result["char_count"])
        cols[2].metric("Слов", result["word_count"])
        cols[3].metric("Ключевых фраз", len(result["keywords"]))

        keywords_df = pd.DataFrame(result["keywords"])
        preview = result["text"][:6000] if result["text"] else "Текст в документе не найден."

        tab_overview, tab_keywords, tab_text, tab_pages = st.tabs(
            ["Обзор", "Ключевые фразы", "Текст", "Страницы"]
        )

        with tab_overview:
            st.write(
                "Документ обработан. Ниже можно скачать JSON-результат или посмотреть "
                "сжатый превью-текст."
            )
            st.caption(f"Тип файла: {result['file_type']}")
            st.download_button(
                label=f"Скачать JSON для {result['file_name']}",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name=f"{result['file_name']}.json",
                mime="application/json",
                key=f"download-{result['file_name']}",
            )
            st.text_area(
                "Короткий просмотр текста",
                value=preview,
                height=240,
                key=f"preview-{result['file_name']}",
            )

        with tab_keywords:
            if keywords_df.empty:
                st.info("Ключевые фразы не найдены.")
            else:
                st.dataframe(keywords_df, use_container_width=True, hide_index=True)
                with st.expander("Показать список отобранных ключевых фраз"):
                    for idx, item in enumerate(result["keywords"], start=1):
                        st.markdown(f"{idx}. `{item['keyword']}`  |  score `{item['score']:.4f}`")

        with tab_text:
            st.text_area(
                "Полный текст",
                value=result["text"] or "Текст не извлечен.",
                height=420,
                key=f"full-{result['file_name']}",
            )

        with tab_pages:
            pages_df = pd.DataFrame(result["pages"])
            st.dataframe(pages_df, use_container_width=True, hide_index=True)

        render_theme_analysis(result)
        render_calculations(result)

    render_dictionary_growth(results)


def render_theme_analysis(result: dict) -> None:
    dictionary_analysis = result.get("dictionary_analysis", {})
    yake_theme_analysis = result.get("yake_theme_analysis", {})
    theme_analysis = result.get("theme_analysis", {})
    prediction = theme_analysis.get("prediction", {})
    yake_prediction = yake_theme_analysis.get("prediction", {})
    label = prediction.get("label", "other")
    yake_label = yake_prediction.get("label", "other")

    st.markdown('<div class="theme-card">', unsafe_allow_html=True)
    st.markdown("### Тематический анализ")
    if label == "gen4" and yake_label == "gen4":
        st.write("Обе проверки считают документ относящимся к тематике реакторов IV поколения.")
    elif label == "gen4" or yake_label == "gen4":
        st.write("Проверки расходятся: одна система видит сильную связь с Generation IV, другая более осторожна.")
    else:
        st.write("Обе проверки считают документ слабо связанным с тематикой реакторов IV поколения.")

    model_badge_class = "decision-good" if label == "gen4" else "decision-bad"
    model_badge_text = "ОТНОСИТСЯ К IV ПОКОЛЕНИЮ" if label == "gen4" else "НЕ ОТНОСИТСЯ К IV ПОКОЛЕНИЮ"
    yake_badge_class = "decision-good" if yake_label == "gen4" else "decision-bad"
    yake_badge_text = "ОТНОСИТСЯ К IV ПОКОЛЕНИЮ" if yake_label == "gen4" else "НЕ ОТНОСИТСЯ К IV ПОКОЛЕНИЮ"
    st.markdown(
        f"""
<div class="decision-grid">
  <div class="decision-card {model_badge_class}">
    <strong>Решение модели</strong>
    <span>{model_badge_text}</span>
  </div>
  <div class="decision-card {yake_badge_class}">
    <strong>Решение YAKE</strong>
    <span>{yake_badge_text}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    model_cols = st.columns(3)
    model_cols[0].metric("Модель: P(gen4)", f"{prediction.get('probability_gen4', 0.0):.2%}")
    model_cols[1].metric("Модель: similarity", f"{prediction.get('similarity', 0.0):.4f}")
    model_cols[2].metric("Вердикт модели", "gen4" if label == "gen4" else "other")

    yake_cols = st.columns(3)
    yake_cols[0].metric("YAKE: P(gen4)", f"{yake_prediction.get('probability_gen4', 0.0):.2%}")
    yake_cols[1].metric("Покрытие словарем", f"{dictionary_analysis.get('coverage_percent', 0.0):.2f}%")
    yake_cols[2].metric("Вердикт YAKE", "gen4" if yake_label == "gen4" else "other")

    with st.expander(" ", expanded=False):
        threshold_cols = st.columns(2)
        threshold_cols[0].metric("Порог модели", f"{prediction.get('threshold', 0.0):.4f}")
        threshold_cols[1].metric("Порог YAKE", f"{yake_prediction.get('threshold', 0.0):.4f}")

        support_cols = st.columns(2)
        support_cols[0].metric("Совпадений словаря", len(dictionary_analysis.get("matched_terms", [])))
        support_cols[1].metric("Совпадений YAKE со словарем", yake_theme_analysis.get("matched_keyword_count", 0))

        matched_terms = dictionary_analysis.get("matched_terms", [])
        if matched_terms:
            st.write("**Совпавшие термины словаря:**")
            st.write(", ".join(matched_terms))
        else:
            st.write("**Совпавшие термины словаря:** пока не найдены.")

        keyword_hits = dictionary_analysis.get("keyword_hits", [])
        if keyword_hits:
            st.write("**Совпадения YAKE со словарем:**")
            hits_df = pd.DataFrame(keyword_hits)
            st.dataframe(hits_df, use_container_width=True, hide_index=True)

    yake_hits = yake_theme_analysis.get("evidence_hits", [])
    if yake_hits:
        st.write("**Признаки, на которые опирался YAKE-вердикт:**")
        yake_df = pd.DataFrame(yake_hits).rename(
            columns={
                "canonical": "Термин словаря",
                "keyword": "Фраза YAKE",
                "score": "YAKE score",
                "strength": "Сила совпадения",
            }
        )
        st.dataframe(yake_df, use_container_width=True, hide_index=True)

    evidence_tokens = theme_analysis.get("evidence_tokens", [])
    if evidence_tokens:
        st.write("**Признаки, на которые опиралась модель:**")
        evidence_df = pd.DataFrame(evidence_tokens).rename(
            columns={
                "token": "Токен/фраза",
                "count": "Частота",
                "direction": "Класс",
                "weight": "Вес",
            }
        )
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)

    suggestions = dictionary_analysis.get("suggestions", [])
    if suggestions and label == "gen4":
        st.write("**Кандидаты на пополнение словаря:**")
        suggestions_df = pd.DataFrame(suggestions).rename(
            columns={"candidate": "Кандидат", "yake_score": "YAKE score"}
        )
        st.dataframe(suggestions_df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_dictionary_growth(results: list[dict]) -> None:
    growth = aggregate_dictionary_growth(results)
    st.markdown("## Пополняемый словарь")
    st.write(
        "Ниже показаны кандидаты на расширение словаря, собранные из документов, "
        "которые модель сочла относящимися к тематике IV поколения."
    )

    cols = st.columns(2)
    cols[0].metric("Принятых тематических документов", growth["accepted_documents"])
    cols[1].metric("Кандидатов на добавление", len(growth["candidate_terms"]))

    if growth["candidate_terms"]:
        growth_df = pd.DataFrame(growth["candidate_terms"]).rename(
            columns={"candidate": "Термин-кандидат", "documents": "Документов"}
        )
        st.dataframe(growth_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Скачать кандидатов словаря",
            data=json.dumps(growth, ensure_ascii=False, indent=2),
            file_name="gen4_dictionary_growth.json",
            mime="application/json",
            key="dictionary-growth-download",
        )
    else:
        st.info("После загрузки тематических документов здесь появятся кандидаты на расширение словаря.")


def render_calculations(result: dict) -> None:
    keywords = result.get("keywords", [])
    with st.expander("Вычисления"):
        st.markdown(
            '<div class="calc-note"><strong>Как читать score:</strong> '
            "в YAKE меньший score означает более сильную и более характерную "
            "ключевую фразу для текущего документа.</div>",
            unsafe_allow_html=True,
        )

        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Фраз найдено", len(keywords))
        if keywords:
            scores = [item["score"] for item in keywords]
            metrics_cols[1].metric("Лучший score", f"{min(scores):.4f}")
            metrics_cols[2].metric("Средний score", f"{sum(scores) / len(scores):.4f}")
        else:
            metrics_cols[1].metric("Лучший score", "нет данных")
            metrics_cols[2].metric("Средний score", "нет данных")

        st.markdown("### Формальная основа")
        st.write(
            "YAKE оценивает отдельные слова по набору статистических признаков, "
            "а затем собирает итоговый вес ключевой фразы из весов слов."
        )

        st.latex(r"WRel = (0.5 + pwl \cdot tf/max\_tf) + (0.5 + pwr \cdot tf/max\_tf)")
        st.latex(r"WFreq = \frac{tf}{avg\_tf + std\_tf}")
        st.latex(r"WSpread = \frac{sentences\_with\_term}{total\_sentences}")
        st.latex(r"WCase = \frac{\max(tf_a, tf_n)}{1 + \log(tf)}")
        st.latex(r"WPos = \log(\log(3 + median\_position))")
        st.latex(
            r"H(term) = \frac{WPos \cdot WRel}{WCase + WFreq/WRel + WSpread/WRel}"
        )
        st.latex(
            r"H(phrase) = \frac{\prod h_i}{(\sum h_i + 1) \cdot tf_{phrase}}"
        )

        st.markdown("### Что означают величины")
        st.write(
            "`tf` показывает частоту терма, `WPos` учитывает раннее появление в тексте, "
            "`WSpread` показывает, в скольких предложениях термин встречается, "
            "`WCase` реагирует на имена собственные и нестандартный регистр, "
            "а `WRel` отражает связи слова с соседями в документе."
        )

        if keywords:
            st.markdown("### Пример на текущем документе")
            top_keywords = pd.DataFrame(keywords).head(8).copy()
            score_min = top_keywords["score"].min()
            score_max = top_keywords["score"].max()

            if score_max == score_min:
                top_keywords["visual_weight"] = 100.0
            else:
                top_keywords["visual_weight"] = (
                    (score_max - top_keywords["score"]) / (score_max - score_min) * 100
                ).round(1)

            top_keywords = top_keywords.rename(
                columns={
                    "keyword": "Ключевая фраза",
                    "score": "Score YAKE",
                    "visual_weight": "Условная сила, %",
                }
            )

            st.write(
                "Ниже показана визуализация найденных фраз. "
                "`Условная сила, %` не является внутренней метрикой YAKE: "
                "это только наглядная шкала, построенная из score для отображения."
            )
            st.bar_chart(
                top_keywords.set_index("Ключевая фраза")["Условная сила, %"],
                color="#f2b84b",
            )
            st.dataframe(top_keywords, use_container_width=True, hide_index=True)

            best_keyword = top_keywords.iloc[0]["Ключевая фраза"]
            best_score = top_keywords.iloc[0]["Score YAKE"]
            st.info(
                f"Например, фраза `{best_keyword}` получила score `{best_score:.4f}`. "
                "Это означает, что внутри данного документа YAKE считает ее одной "
                "из наиболее характерных и информативных."
            )
        else:
            st.info("Для этого документа YAKE не вернул ключевые фразы, поэтому пример расчета не отображается.")


def _parse_variants(raw_value: str, canonical: str) -> list[str]:
    parts = [
        item.strip()
        for chunk in raw_value.replace(";", ",").split(",")
        for item in [chunk]
        if item.strip()
    ]
    variants = [canonical.strip()]
    for part in parts:
        if part and part.lower() not in {item.lower() for item in variants}:
            variants.append(part)
    return variants


def render_dictionary_editor_target(target: str, title: str) -> None:
    active_dictionary = load_seed_dictionary(target=target)
    overrides = load_manual_dictionary_overrides()
    target_overrides = overrides["targets"].get(target, {"blocked_terms": [], "manual_entries": []})
    manual_entries = target_overrides.get("manual_entries", [])
    blocked_terms = target_overrides.get("blocked_terms", [])

    editor_key = f"dictionary_editor_rows_{target}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = [
            {
                "canonical": item.get("canonical", ""),
                "variants_csv": ", ".join(item.get("variants", [])),
                "weight": float(item.get("weight", 2.0)),
            }
            for item in manual_entries
        ]

    st.markdown(f"### {title}")
    metrics = st.columns(3)
    metrics[0].metric("Активных терминов", len(active_dictionary.get("entries", [])))
    metrics[1].metric("Ручных терминов", len(manual_entries))
    metrics[2].metric("Исключенных терминов", len(blocked_terms))

    st.caption("Можно добавить свои термины, а также исключить ненужные из активного словаря.")

    editor_df = pd.DataFrame(
        st.session_state[editor_key],
        columns=["canonical", "variants_csv", "weight"],
    )
    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key=f"manual-dictionary-editor-{target}",
        column_config={
            "canonical": st.column_config.TextColumn("Термин"),
            "variants_csv": st.column_config.TextColumn("Варианты через запятую"),
            "weight": st.column_config.NumberColumn("Вес", min_value=0.1, max_value=10.0, step=0.1),
        },
    )
    st.session_state[editor_key] = edited_df.to_dict("records")

    if st.button(f"Сохранить словарь {title.lower()}", key=f"save-dictionary-{target}", use_container_width=True):
        cleaned_entries = []
        seen_terms: set[str] = set()
        for row in st.session_state[editor_key]:
            canonical = str(row.get("canonical", "")).strip()
            if not canonical or canonical.lower() in seen_terms:
                continue
            seen_terms.add(canonical.lower())
            cleaned_entries.append(
                {
                    "canonical": canonical,
                    "variants": _parse_variants(str(row.get("variants_csv", "")), canonical),
                    "weight": float(row.get("weight", 2.0) or 2.0),
                    "category": f"manual_{target}",
                }
            )
        payload = load_manual_dictionary_overrides()
        payload["targets"][target]["manual_entries"] = cleaned_entries
        save_manual_dictionary_overrides(payload)
        reset_model_cache()
        st.success(f"Ручной словарь для {title.lower()} сохранен.")
        st.rerun()

    runtime_dictionary_path = PROJECT_DIR / "data" / "runtime_dictionary.json"
    if runtime_dictionary_path.exists():
        base_payload = json.loads(runtime_dictionary_path.read_text(encoding="utf-8"))
    else:
        base_payload = load_seed_dictionary(target=target)
    base_terms = sorted(
        set(blocked_terms)
        | {
            item["canonical"]
            for item in base_payload.get("entries", [])
        }
    )
    blocked_selection = st.multiselect(
        f"Исключить термины из словаря {title.lower()}",
        options=base_terms,
        default=blocked_terms,
        key=f"blocked-terms-{target}",
    )
    if st.button(f"Сохранить исключения для {title.lower()}", key=f"save-blocked-{target}", use_container_width=True):
        payload = load_manual_dictionary_overrides()
        payload["targets"][target]["blocked_terms"] = blocked_selection
        save_manual_dictionary_overrides(payload)
        reset_model_cache()
        st.success(f"Исключения для {title.lower()} сохранены.")
        st.rerun()


def render_dictionary_editors() -> None:
    st.markdown("## Редактирование словарей")
    st.write(
        "В самом низу можно вручную дополнять или чистить словарь отдельно для нашей модели и отдельно для YAKE."
    )
    with st.expander("Редактировать словарь модели", expanded=False):
        render_dictionary_editor_target("model", "Словарь модели")
    with st.expander("Редактировать словарь YAKE", expanded=False):
        render_dictionary_editor_target("yake", "Словарь YAKE")


def render_visual_dashboard() -> None:
    base_rows = get_base_overview()
    test_rows = get_test_overview()
    model_eval = evaluate_theme_model()
    model_dictionary = load_seed_dictionary(target="model")
    yake_dictionary = load_seed_dictionary(target="yake")
    overrides = load_manual_dictionary_overrides()
    runtime_summary = st.session_state.get("runtime_summary", {})

    active_rows = [row for row in base_rows if row.get("include") and row.get("label") != "skip"]
    gen4_count = sum(1 for row in active_rows if row.get("label") == "gen4")
    other_count = sum(1 for row in active_rows if row.get("label") == "other")
    skip_count = sum(1 for row in base_rows if row.get("label") == "skip" or not row.get("include"))
    manual_model_count = len(overrides["targets"].get("model", {}).get("manual_entries", []))
    manual_yake_count = len(overrides["targets"].get("yake", {}).get("manual_entries", []))
    current_results = st.session_state.get("results", [])

    st.markdown("## Визуализация")
    st.markdown(
        f"""
<section class="visual-shell">
  <span class="visual-kicker">Снимок системы</span>
  <h3>Визуальный блок текущего состояния модели и словарей</h3>
  <p>
    Здесь в наглядном виде собраны размеры корпуса, активные словари, качество на test
    и путь прохождения документа через извлекатель, YAKE и нашу тематическую модель.
  </p>
  <div class="visual-grid">
    <div class="visual-card">
      <strong>Документов в base</strong>
      <span class="visual-number">{len(base_rows)}</span>
      <span class="visual-caption">Эталонный корпус Generation IV</span>
    </div>
    <div class="visual-card">
      <strong>Активно в обучении</strong>
      <span class="visual-number">{len(active_rows)}</span>
      <span class="visual-caption">gen4: {gen4_count} | other: {other_count}</span>
    </div>
    <div class="visual-card">
      <strong>Точность на test</strong>
      <span class="visual-number">{model_eval["accuracy_percent"]}%</span>
      <span class="visual-caption">Документов в проверке: {len(test_rows)}</span>
    </div>
    <div class="visual-card">
      <strong>Runtime-словарь</strong>
      <span class="visual-number">{len(model_dictionary.get("entries", []))}</span>
      <span class="visual-caption">Цель сборки: {runtime_summary.get("dictionary_entries", len(model_dictionary.get("entries", [])))}</span>
    </div>
  </div>
  <div class="visual-pipeline">
    <div class="pipeline-step">
      <strong>1. Загрузка</strong>
      <span>PDF, DOC и DOCX поступают в сервис через веб-форму.</span>
    </div>
    <div class="pipeline-step">
      <strong>2. Извлечение</strong>
      <span>PyMuPDF и LibreOffice переводят документ в чистый текст.</span>
    </div>
    <div class="pipeline-step">
      <strong>3. YAKE</strong>
      <span>Выделяются ключевые фразы и считается независимый YAKE-вердикт.</span>
    </div>
    <div class="pipeline-step">
      <strong>4. Наша модель</strong>
      <span>Считается similarity к корпусу IV поколения по словарю домена.</span>
    </div>
    <div class="pipeline-step">
      <strong>5. Решение</strong>
      <span>На странице показываются обе оценки принадлежности документа.</span>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

    top_cols = st.columns(4)
    top_cols[0].metric("Словарь модели", len(model_dictionary.get("entries", [])))
    top_cols[1].metric("Словарь YAKE", len(yake_dictionary.get("entries", [])))
    top_cols[2].metric("Ручных терминов модели", manual_model_count)
    top_cols[3].metric("Ручных терминов YAKE", manual_yake_count)

    chart_cols = st.columns(2)
    distribution_df = pd.DataFrame(
        [
            {"Статус": "gen4", "Документов": gen4_count},
            {"Статус": "other", "Документов": other_count},
            {"Статус": "skip", "Документов": skip_count},
        ]
    )
    chart_cols[0].markdown("### Распределение корпуса")
    chart_cols[0].bar_chart(distribution_df.set_index("Статус"), color="#f2b84b")

    confusion_df = pd.DataFrame(
        [{"Переход": key, "Количество": value} for key, value in model_eval["confusion"].items()]
    )
    chart_cols[1].markdown("### Проверка модели на test")
    chart_cols[1].bar_chart(confusion_df.set_index("Переход"), color="#93d6b5")

    if current_results:
        summary_rows = []
        for item in current_results:
            summary_rows.append(
                {
                    "Документ": item["file_name"],
                    "Наша модель": item.get("theme_analysis", {}).get("prediction", {}).get("label", "other"),
                    "YAKE": item.get("yake_theme_analysis", {}).get("prediction", {}).get("label", "other"),
                    "Similarity": item.get("theme_analysis", {}).get("prediction", {}).get("similarity", 0.0),
                    "P(gen4)": item.get("theme_analysis", {}).get("prediction", {}).get("probability_gen4", 0.0),
                }
            )
        st.markdown("### Последние документы в текущей сессии")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def ensure_runtime_bootstrap() -> None:
    if st.session_state.get("runtime_bootstrap_done"):
        return
    summary = build_runtime_resources(force_cache=False)
    reset_model_cache()
    reset_dictionary_cache()
    st.session_state["runtime_bootstrap_done"] = True
    st.session_state["runtime_summary"] = summary


def render_training_lab() -> None:
    st.markdown("## Лаборатория обучения")
    st.write(
        "Здесь используется эталонный корпус из папки `base`. Из него строятся "
        "300 терминов словаря Generation IV и корпусная модель похожести."
    )

    base_rows = get_base_overview()
    base_map = {row["file_name"]: row for row in load_base_labels()}

    if "training_editor_rows" not in st.session_state:
        st.session_state["training_editor_rows"] = base_rows

    editor_df = pd.DataFrame(st.session_state["training_editor_rows"])
    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        key="base-training-editor",
        column_config={
            "include": st.column_config.CheckboxColumn("Включить"),
            "label": st.column_config.SelectboxColumn(
                "Метка",
                options=["gen4", "other", "skip"],
            ),
            "file_name": st.column_config.TextColumn("Файл", disabled=True),
            "file_type": st.column_config.TextColumn("Тип", disabled=True),
            "language": st.column_config.TextColumn("Язык", disabled=True),
            "pages": st.column_config.NumberColumn("Страниц", disabled=True),
            "words": st.column_config.NumberColumn("Слов", disabled=True),
            "note": st.column_config.TextColumn("Комментарий"),
            "preview": st.column_config.TextColumn("Превью", disabled=True, width="large"),
        },
    )
    st.session_state["training_editor_rows"] = edited_df.to_dict("records")

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Обновить кэш base", use_container_width=True):
        with st.spinner("Переизвлекаю тексты из документов base..."):
            build_base_cache(force=True)
            st.session_state["training_editor_rows"] = get_base_overview()
            st.session_state["runtime_bootstrap_done"] = False
            ensure_runtime_bootstrap()
        st.success("Кэш base обновлен.")

    if col_b.button("Сбросить таблицу", use_container_width=True):
        st.session_state["training_editor_rows"] = get_base_overview()

    if col_c.button("Переобучить модель и словарь", use_container_width=True):
        edited_rows = st.session_state.get("training_editor_rows", base_rows)
        save_base_labels(
            [
                {
                    "file_name": row["file_name"],
                    "path": base_map[row["file_name"]]["path"],
                    "label": row["label"],
                    "include": row["include"],
                    "note": row.get("note", ""),
                }
                for row in edited_rows
            ]
        )
        with st.spinner("Переобучаю модель по корпусу base и пересобираю словарь..."):
            build_runtime_resources(force_cache=False)
            reset_model_cache()
            reset_dictionary_cache()
            st.session_state["training_editor_rows"] = get_base_overview()
        st.success("Модель и словарь пересобраны по текущей разметке.")

    labels_df = pd.DataFrame(st.session_state["training_editor_rows"])
    active_df = labels_df[(labels_df["include"]) & (labels_df["label"] != "skip")]
    gen4_count = int((active_df["label"] == "gen4").sum())
    other_count = int((active_df["label"] == "other").sum())
    model_eval = evaluate_theme_model()
    dictionary = load_seed_dictionary()

    metrics = st.columns(4)
    metrics[0].metric("Файлов в base", len(labels_df))
    metrics[1].metric("В обучении gen4", gen4_count)
    metrics[2].metric("В обучении other", other_count)
    metrics[3].metric("Точность на test", f"{model_eval['accuracy_percent']}%")

    with st.expander("Просмотр текущей модели"):
        confusion_df = pd.DataFrame(
            [{"Переход": key, "Количество": value} for key, value in model_eval["confusion"].items()]
        )
        st.dataframe(confusion_df, use_container_width=True, hide_index=True)
        st.write(f"Словарь сейчас содержит {len(dictionary.get('entries', []))} записей.")

        cache_rows = {item["file_name"]: item for item in build_base_cache(force=False)["documents"]}
        preview_rows = []
        for row in st.session_state["training_editor_rows"]:
            if not row["include"] or row["label"] == "skip":
                continue
            cached = cache_rows.get(row["file_name"])
            if not cached:
                continue
            prediction = analyze_text_theme(cached["text"])
            preview_rows.append(
                {
                    "Файл": row["file_name"],
                    "Разметка": row["label"],
                    "Предсказание": prediction["prediction"]["label"],
                    "P(gen4)": prediction["prediction"]["probability_gen4"],
                }
            )
        if preview_rows:
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    with st.expander("Список отобранных терминов словаря"):
        dictionary_rows = pd.DataFrame(dictionary.get("entries", [])).copy()
        if dictionary_rows.empty:
            st.info("Словарь пока пуст.")
        else:
            display = dictionary_rows.rename(
                columns={
                    "canonical": "Термин",
                    "doc_count": "Документов",
                    "avg_score": "Средний YAKE score",
                }
            )
            keep_columns = [col for col in ["Термин", "Документов", "Средний YAKE score"] if col in display.columns]
            st.dataframe(display[keep_columns], use_container_width=True, hide_index=True, height=360)
            with st.expander("Показать термины списком"):
                for idx, term in enumerate(display["Термин"].tolist(), start=1):
                    st.markdown(f"{idx}. `{term}`")

    with st.expander("Проверочная выборка test"):
        test_rows = get_test_overview()
        if test_rows:
            st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Папка `test` пока не найдена или в ней нет файлов.")


def main() -> None:
    ensure_runtime_bootstrap()
    render_hero()
    language, max_keywords, model_threshold, yake_threshold = render_sidebar()
    render_training_lab()

    with st.form("upload-form"):
        st.markdown("## Загрузка документов")
        uploads = st.file_uploader(
            "Перетащите PDF, DOC или DOCX сюда или выберите их вручную",
            type=["pdf", "doc", "docx"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Извлечь данные")

    if submitted:
        if not uploads:
            st.warning("Добавьте хотя бы один файл.")
            st.session_state["results"] = []
            st.session_state["errors"] = []
        else:
            with st.spinner("Извлекаю текст и ключевые фразы..."):
                results, errors = process_files(
                    uploads,
                    language=language,
                    max_keywords=max_keywords,
                    model_threshold=model_threshold,
                    yake_threshold=yake_threshold,
                )
            st.session_state["results"] = results
            st.session_state["errors"] = errors

    for error in st.session_state.get("errors", []):
        st.error(error)

    current_results = apply_theme_analyses(
        st.session_state.get("results", []),
        model_threshold=model_threshold,
        yake_threshold=yake_threshold,
    )
    st.session_state["results"] = current_results
    render_results(current_results)
    render_dictionary_editors()
    render_visual_dashboard()


if __name__ == "__main__":
    main()
