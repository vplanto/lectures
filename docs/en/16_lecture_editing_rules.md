# Lecture Editing Rules

> 🇺🇦 [Українська версія](../16_lecture_editing_rules.md) | 🇬🇧 English version

Canonical rules for **Ukrainian lecture prose** live in the Ukrainian document. This page summarizes scope and links for collaborators editing `docs/**`.

---

## Purpose

Complements [AI lecture protocol (10)](../10_ai_lectures.md) (process, volume, QA) with **style and presentation** rules when rewriting existing lectures.

**Core rule:** rewriting ≠ shortening. Change **delivery** (hook, rhythm, hierarchy), not **educational content**.

---

## Scope

| Applies to | Excluded |
|------------|----------|
| `NN_*.md`, `lecture_*.md`, seminars in `docs/**` | NMK: `methodology.md`, `sources.md`, `DISCLAIMER.md` |
| | `index.md`, `study_materials.md`, `material_study.md` |

---

## Key constraints (summary)

1. **Anti-LLM, not telegraphese** — no template intros/conclusions; start *in medias res*; paragraphs ≤ 4 lines (split, don't delete).
2. **Volume** — ≥ ~90% of original prose; keep cases, `>` notes, formulas, simulators, exam blocks.
3. **Language** — Ukrainian prose; English only for names, established acronyms, code; explain terms on first mention.
4. **Pedagogy** — hook ≠ mini-case; no questions about material not yet introduced.
5. **Minimal diff** — no git commit unless explicitly requested.

Full checklist and examples: **[Українська версія](../16_lecture_editing_rules.md)**.

---

## Related

- [10_ai_lectures.md](../10_ai_lectures.md) — generation workflow
- [DISCLAIMER.md](../DISCLAIMER.md) — academic integrity
- [methodology.md](./methodology.md) — hub methodology map
