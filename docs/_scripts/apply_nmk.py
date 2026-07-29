#!/usr/bin/env python3
"""Apply NMK: prepend disclaimers, add NMK block to index files, optional YAML."""
from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent

DISCLAIMER_UA = (
    "> **Академічна доброчесність.** Матеріали відповідають вимогам "
    "[Закону України № 4742-IX]({disclaimer}). Використання ШІ — "
    "[протокол]({ai}). Оцінювання — [Risk & Reward]({grading}). "
    "Джерела курсу: [sources.md]({sources}).\n\n"
)

DISCLAIMER_EN = (
    "> **Academic Integrity.** Materials comply with "
    "[Ukrainian Law No. 4742-IX]({disclaimer}). AI use — "
    "[protocol]({ai}). Grading — [Risk & Reward]({grading}). "
    "Course sources: [sources.md]({sources}).\n\n"
)

NMK_BLOCK_UA = """
### Навчально-методичний комплект (НМК)

| | |
|---|---|
| **Методологія** | [Траєкторія курсу]({methodology}) |
| **Джерела** | [Реєстр першоджерел]({sources}) |
| **Декларація** | [Академічна доброчесність]({disclaimer}) |

"""

NMK_BLOCK_EN = """
### Teaching Kit (NMK)

| | |
|---|---|
| **Methodology** | [Course trajectory]({methodology}) |
| **Sources** | [Primary-source register]({sources}) |
| **Declaration** | [Academic integrity]({disclaimer}) |

"""

# (course_dir_relative_to_docs, exclude_filenames, en_subdir or None)
COURSES = [
    ("gdl_in_logistics", {
        "exclude": {"index.md", "study_materials.md", "methodology.md", "sources.md"},
        "en": "en",
    }),
    ("nlp_signal_noise", {
        "exclude": {"index.md", "study_materials.md", "methodology.md", "sources.md"},
        "en": "en",
    }),
    ("observability_distributed_systems", {
        "exclude": {"index.md", "material_study.md", "methodology.md", "sources.md"},
        "en": None,
    }),
    ("prescriptive_xai_optimization", {
        "exclude": {"index.md", "material_study.md", "methodology.md", "sources.md", "README_code.md"},
        "en": None,
    }),
    ("predictive_chaos_monitoring", {
        "exclude": {"index.md", "material_study.md", "methodology.md", "sources.md", "README.md", "side_notes_advanced.md"},
        "en": None,
    }),
    ("predictive", {
        "exclude": {"index.md", "methodology.md", "sources.md"},
        "en": None,
    }),
    ("semantic_decoder", {
        "exclude": {"index.md", "methodology.md", "sources.md"},
        "en": None,
    }),
    ("k8s_llm", {
        "exclude": {"index.md", "methodology.md", "sources.md"},
        "en": "en",
    }),
    ("urban_evac_sim", {
        "exclude": {"index.md", "methodology.md", "sources.md"},
        "en": None,
    }),
]

MODULE_HINTS = {
    "gdl_in_logistics": [
        (0, "Вступ"), (1, "Фундамент"), (2, "Фундамент"),
        (3, "AlphaFold 2"), (4, "AlphaFold 2"), (5, "AlphaFold 2"),
        (6, "Синтез"), (7, "Синтез"), (8, "Практикум"), (9, "Практикум"),
        (10, "Практикум"), (11, "Семінар"), (12, "Семінар"), (13, "Семінар"),
    ],
    "nlp_signal_noise": [
        (0, "Інтуїція"), (1, "Інтуїція"), (2, "Статистика"), (3, "Статистика"),
        (4, "Семантика"), (5, "Семантика"), (6, "Практикум"), (7, "Практикум"),
        (8, "Практикум"), (9, "Семінар"), (10, "Семінар"), (11, "Семінар"),
        (12, "Семінар"), (13, "Глосарій"),
    ],
}

LECTURE_TYPE = {
    "seminar": re.compile(r"seminar|SEMINARS_README", re.I),
    "project": re.compile(r"^08_final_project|^09_scope", re.I),
}


def hub_paths(depth: int, lang: str) -> tuple[str, str, str, str]:
    """disclaimer, ai, grading, sources relative paths from file at depth."""
    prefix = "../" * depth
    if lang == "en":
        return (
            f"{prefix}en/DISCLAIMER.md",
            f"{prefix}en/10_ai_lectures.md",
            f"{prefix}en/06_grading_experiment.md",
            "./sources.md" if depth <= 1 else f"{prefix}sources.md",
        )
    return (
        f"{prefix}DISCLAIMER.md",
        f"{prefix}10_ai_lectures.md",
        f"{prefix}06_grading_experiment.md",
        "./sources.md" if depth <= 1 else f"{prefix}sources.md",
    )


def make_disclaimer(depth: int, lang: str) -> str:
    d, ai, g, s = hub_paths(depth, lang)
    tmpl = DISCLAIMER_EN if lang == "en" else DISCLAIMER_UA
    return tmpl.format(disclaimer=d, ai=ai, grading=g, sources=s)


def has_disclaimer(text: str) -> bool:
    return "Академічна доброчесність" in text or "Academic Integrity" in text


def insert_after_frontmatter(text: str, block: str) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[: end + 3] + "\n\n" + block + text[end + 3 :].lstrip("\n")
    return block + text


def prepend_disclaimer(path: Path, depth: int, lang: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if has_disclaimer(text):
        return False
    block = make_disclaimer(depth, lang)
    new_text = insert_after_frontmatter(text, block)
    path.write_text(new_text, encoding="utf-8")
    return True


def infer_yaml(path: Path, course: str) -> dict | None:
    name = path.name
    m = re.match(r"^(\d+)[b]?_", name)
    if not m and not name.startswith("lecture_"):
        if "SEMINARS" in name:
            return {"title": "Seminars Overview", "type": "seminar", "module": "Workshop"}
        return None
    num = int(m.group(1)) if m else 0
    hints = MODULE_HINTS.get(course, [])
    module = next((mod for n, mod in hints if n == num), f"Module {num}")
    ltype = "seminar" if LECTURE_TYPE["seminar"].search(name) else "lecture"
    if LECTURE_TYPE["project"].search(name):
        ltype = "project"
    title = name.replace(".md", "").replace("_", " ").title()
    prereq = f"module {num - 1}" if num > 0 else "none"
    return {
        "title": title,
        "type": ltype,
        "module": module,
        "prerequisites": prereq,
        "layout": "default",
    }


def ensure_yaml(path: Path, course: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        fm = text.split("---", 2)[1]
        if "type:" in fm:
            return False
    meta = infer_yaml(path, course)
    if not meta:
        return False
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f'{k}: "{v}"' if k == "title" else f"{k}: {v}")
    lines.append("---\n")
    if text.startswith("---"):
        end = text.find("---", 3)
        body = text[end + 3 :].lstrip("\n")
    else:
        body = text
    path.write_text("\n".join(lines) + "\n" + body, encoding="utf-8")
    return True


def add_nmk_to_index(index_path: Path, lang: str, rel_methodology: str, rel_sources: str, rel_disclaimer: str) -> bool:
    text = index_path.read_text(encoding="utf-8")
    if "НМК" in text or "Teaching Kit (NMK)" in text:
        return False
    block_tmpl = NMK_BLOCK_EN if lang == "en" else NMK_BLOCK_UA
    block = block_tmpl.format(
        methodology=rel_methodology,
        sources=rel_sources,
        disclaimer=rel_disclaimer,
    )
    # Insert after first --- block or after audience/metadata section
    markers = [
        "\n---\n\n### Структура",
        "\n---\n\n### Анотація",
        "\n---\n\n## ",
        "\n---\n\n### Course Structure",
        "\n---\n\n## Structure",
        "\n---\n\n##  Структура",
    ]
    for marker in markers:
        if marker in text:
            text = text.replace(marker, block + marker, 1)
            index_path.write_text(text, encoding="utf-8")
            return True
    # fallback: after second ---
    parts = text.split("---", 2)
    if len(parts) >= 3:
        new_text = parts[0] + "---" + parts[1] + "---" + block + parts[2]
        index_path.write_text(new_text, encoding="utf-8")
        return True
    return False


def process_course(course: str, cfg: dict) -> None:
    base = DOCS / course
    exclude = cfg["exclude"]

    # UA index
    idx = base / "index.md"
    if idx.exists():
        add_nmk_to_index(idx, "uk", "./methodology.md", "./sources.md", "../DISCLAIMER.md")

    # EN index
    if cfg.get("en"):
        en_idx = base / cfg["en"] / "index.md"
        if en_idx.exists():
            add_nmk_to_index(en_idx, "en", "./methodology.md", "../sources.md", "../../en/DISCLAIMER.md")

    # UA lectures
    for md in sorted(base.glob("*.md")):
        if md.name in exclude:
            continue
        if prepend_disclaimer(md, 1, "uk"):
            print(f"  disclaimer UA: {md.relative_to(DOCS)}")
        if re.match(r"^\d+[_b]?", md.name) or md.name.startswith("lecture_") or "SEMINARS" in md.name:
            if ensure_yaml(md, course):
                print(f"  yaml: {md.relative_to(DOCS)}")

    # EN lectures
    if cfg.get("en"):
        en_dir = base / cfg["en"]
        for md in sorted(en_dir.glob("*.md")):
            if md.name in exclude or md.name in ("methodology.md", "study_materials.md", "index.md"):
                continue
            if prepend_disclaimer(md, 2, "en"):
                print(f"  disclaimer EN: {md.relative_to(DOCS)}")


def centralize_bibliography(path: Path, course: str) -> bool:
    """Replace trailing '### Академічні' / '## Джерела' sections with pointer to sources.md."""
    text = path.read_text(encoding="utf-8")
    patterns = [
        r"\n### Академічні Статті\n[\s\S]*$",
        r"\n## Джерела та література\n[\s\S]*$",
        r"\n## References\n[\s\S]*$",
    ]
    replacement = (
        "\n\n---\n\n**Джерела:** повний реєстр `[1]…[N]` — [sources.md](./sources.md).\n"
    )
    for pat in patterns:
        if re.search(pat, text):
            new_text = re.sub(pat, replacement, text)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                return True
    return False


def main() -> None:
    print("=== NMK apply ===")
    for course, cfg in COURSES:
        print(f"\n[{course}]")
        process_course(course, cfg)

    # Centralize bibliographies in pilot lectures
    for f in [
        DOCS / "nlp_signal_noise" / "00_the_bayesian_trap.md",
        DOCS / "gdl_in_logistics" / "01_classical_transport_problem.md",
    ]:
        if f.exists() and centralize_bibliography(f, f.parent.name):
            print(f"  sources ref: {f.relative_to(DOCS)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
