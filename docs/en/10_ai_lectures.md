# Protocol for Developing High-Volume Technical Lectures (16k+ characters)

> 🇺🇦 [Українська версія](../10_ai_lectures.md) | 🇬🇧 English version

This document defines standard for iterative development of educational materials to prevent technical density degradation and volume loss due to LLM limitations (RLHF-brevity and Attention Drift).

**Style and presentation** when rewriting finished lectures — see [Lecture Editing Rules (16)](../16_lecture_editing_rules.md) (not duplicated here).

---

## 0. Limitations Diagnosis (Root Cause)

1. **RLHF-brevity:** Models are trained to provide concise answers. When requesting 16k characters, model is prone to "structure hallucinations" — it pretends to write a lot using general phrases.
2. **Context Window vs Output Token Limit:** Even with 200k+ context, single output limit is usually restricted to 4k-8k tokens.
3. **Model laziness:** When editing large blocks, LLM often uses placeholders `[text remains unchanged]`, destroying document integrity.

---

## Stage 1: Architectural Design (Technical Skeleton)

Goal: Create rigid matrix preventing model from deviating from specified volume and depth.

### 1.1. Volume Distribution Matrix
Each section must have clear character limit. Example for 16,000 character lecture:

* **Section 1: Introduction and Problem Statement (~1,500 chars)**
    * Context (High-load/Performance).
    * Theses and Core Thesis.
* **Section 2: Theoretical Foundation (~4,500 chars)**
    * Deep algorithm/architecture analysis.
    * Mathematical justification (LaTeX).
* **Section 3: Practical Implementation (~6,000 chars)**
    * Code (ready for copy-paste).
    * Optimization and FinOps (resource consumption metrics).
* **Section 4: Validation and Cases (~2,500 chars)**
    * Benchmarks and Edge Cases.
* **Section 5: Conclusions and Checklist (~1,500 chars)**

### 1.2. Technical Requirements (Constraints)
* **No Generalities:** Prohibited to use basic term definitions (K8s, O-notation). Only implementation specifics.
* **Stack-Specific:** Mandatory version fixation (e.g., Python 3.12, PostgreSQL 16).
* **Style:** Expert-to-Expert (pragmatic, without epithets).

---

## Stage 2: "Atomic Write" Protocol (Incremental Generation)

Goal: Bypass output token limit through step-by-step assembly.

### 2.1. Buffering Rule (4k Tokens)
Entire process is broken into **Stages**. One request = one logical block (up to 4000 characters).

**Action algorithm:**
1. Provide AI "Technical Skeleton".
2. Command: `Generate exclusively Section 1. Adhere to 1500 character limit. Stop after completion`.
3. Command: `Generate Section 2. Use Section 1 context, but output ONLY new text`.

### 2.2. Strict Output Control
When adding new blocks use instruction:
> **APPEND MODE:** Output only new sections. Categorically forbidden to use `...` or abbreviate existing text. If volume exceeds output limit — stop and request continuation confirmation.

---

## Stage 3: Finalization and Quality Assurance (QA)

### 3.1. Markup and Formula Validation
* LaTeX check: All formulas must be in $inline$ or $$display$$ (render check).
* Code check: No abbreviations inside functions.
* Markdown control: Correct header hierarchy (H1 -> H2 -> H3).

### 3.2. Assembly into Markdown Block
For copying convenience, entire final document (or large block) must be wrapped in code block:
```markdown
[Entire lecture text here]

```

---

## Stage 4: Tooling (Agents & IDE)

For working with documents of this volume, recommended tools with direct file system access:

1. **Cursor (Composer Mode):** * Use `@file` for context.
* Prompt: `Update section 3 in @lecture.md. Expand implementation details to 3000 symbols. Keep other sections intact.`
2. **Claude Code (CLI):** * Allows edits via `diff`, guaranteeing preservation of remaining text without regeneration.
3. **Obsidian / VS Code:**
* For local assembly of blocks generated through chat interface.

---

## Checklist for launching new lecture:

1. [ ] Topic and target audience formed (Senior/Lead).
2. [ ] Skeleton created with character distribution.
3. [ ] Technology stack defined.
4. [ ] "Atomic Write" mode activated (block-by-block generation).

```
