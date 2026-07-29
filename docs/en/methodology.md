# Portal Methodology

> 🇺🇦 [Українська версія](../methodology.md) | 🇬🇧 English version

This document describes the **role of the central hub** `lectures/docs/` and how methodological articles connect. Assessment details and the AI protocol are in separate documents (linked below) — **their content is not duplicated here**.

---

## 1. Portal Purpose

The portal unifies:

1. **Disciplinary courses** (external GitHub Pages: Web, Blockchain, C++, Java) — links only in [`index.md`](./index.md).
2. **Local courses and research tracks** — full content in subfolders (`gdl_in_logistics/`, `nlp_signal_noise/`, etc.).
3. **Methodological work** — articles 00–15 in the `docs/` root.
4. **Active Projects** — `k8s_llm/`, `urban_evac_sim/`.

Each local course has its own [`methodology.md`](../gdl_in_logistics/methodology.md) with module trajectory; this file covers **hub level only**.

---

## 2. Methodological Articles Map (00–15)

| # | Document | Focus |
|---|----------|-------|
| 00 | [Modernizing Materials](../00_method_changes.md) | From slides to living Markdown practicums |
| 01 | [R&M Activities 2025/26](../01_2025_science_education.md) | Practical student training |
| 02 | [Research Directions](../02_research.md) | Research context for courses |
| 03 | [Path to Personal AI](../03_graal.md) | From «creator» to «architect» |
| 04 | [NotebookLM](../04_notebooklm_manual.md) | Deep Research, audio overviews |
| 05 | [AI as Socratic Tutor](../05_ai_socratic_tutor.md) | Harvard Tutor on Java course |
| 06 | [**Risk & Reward Grading**](../06_grading_experiment.md) | Cumulative model + optional exam |
| 07 | [Epsilon-Greedy](../07_mental_overfitting.md) | Avoiding mental overfitting |
| 08 | Infrastructure barrier *(file in progress)* | Infrastructure Literacy |
| 09 | [Experience Architecture](../09_experience_architecture.md) | Knowledge compression → expertise |
| 10 | [**AI Lecture Protocol**](../10_ai_lectures.md) | Atomic Write, QA, technical density |
| 11 | [AI Landscape 2026](../11_ai_landscape_2026.md) | Model and mode taxonomy |
| 12 | [Accreditation Mapping](../12_accreditation_mapping.md) | Standard 113 ↔ courses |
| 13 | [Higher Ed vs Vocational](../13_higher_education_vs_vocational.md) | Scientific method in universities |
| 14 | [Specialist vs Generalist](../14_specialist_generalist_dilemma.md) | π-shaped, Centaur model |
| 15 | [Article Structure](../15_academic_article_structure.md) | IMRaD, hourglass model |
| 16 | [**Lecture Editing Rules**](../16_lecture_editing_rules.md) | Anti-LLM style, volume, language, checklist |

---

## 3. Course-Level Teaching Kit (NMK)

For each local subfolder with `index.md`:

| Artifact | Purpose |
|----------|---------|
| [`DISCLAIMER.md`](../DISCLAIMER.md) (hub) | Full text; short insert in lectures |
| `{course}/methodology.md` | Module trajectory, links to `study_materials` / `material_study` |
| `{course}/sources.md` | Primary-source register `[1]…[N]` |
| `{course}/exam.md` | Optional — only if the course has an exam pool |

---

## 4. Related Regulations (do not duplicate)

- **Assessment:** [06_grading_experiment.md](../06_grading_experiment.md)
- **AI-assisted development:** [10_ai_lectures.md](../10_ai_lectures.md)
- **Lecture rewrite style:** [16_lecture_editing_rules.md](../16_lecture_editing_rules.md)
- **Academic integrity:** [DISCLAIMER.md](../DISCLAIMER.md)
- **Hub sources:** [sources.md](./sources.md)
