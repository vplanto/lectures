# Dataset Engineering Kit: Kubernetes LLM Fine-Tuning

> 🇺🇦 [Українська версія](../index.md) | 🇬🇧 English version

This repository contains complete set of instructions, standards, and tools for creating specialized dataset and training model.
**Project goal:** Train Large Language Model (LLM) to perform **SRE assistant** role for failure diagnostics and FinOps optimization.

### Teaching Kit (NMK)

| | |
|---|---|
| **Methodology** | [Course trajectory](./methodology.md) |
| **Sources** | [Primary-source register](../sources.md) |
| **Declaration** | [Academic integrity](../../en/DISCLAIMER.md) |


---

##  Documentation Structure

### 1. [01_k8s_problem_taxonomy.md](./01_k8s_problem_taxonomy.md) — Theory
**"What are we looking for?"**
Fundamental document defining model knowledge **scope**. Classifies problems into "Break-fix" and "Optimization".

### 2. [02_detection_and_symptoms.md](./02_detection_and_symptoms.md) — Diagnostics
**"What does it look like?"**
Reference guide for technical markers (`kubectl` commands, Prometheus metrics) for identifying problems in logs.

### 3. [03_dataset_output_templates.md](./03_dataset_output_templates.md) — Markup
**"How should model respond?"**
Collection of "Gold Standard" JSON responses. Contains formulas for **quantitative recommendations** (P95 + Buffer).

### 4. [04_data_preparation_workflow.md](./04_data_preparation_workflow.md) — Data Engineering
**"How to prepare data?"**
Obfuscation instruction (security) and **synthesis** (Error Injection) for creating balanced dataset.

### 5. [05_model_lifecycle_and_metrics.md](./05_model_lifecycle_and_metrics.md) — Validation
**"How to verify result?"**
Quality assessment methodology. Describes metrics (JSON Validity, Math Consistency) and testing process before release.

---

##  Working Data (Dataset)

* **`obfuscator.py`**: Script for cleaning sensitive data. (Repository contains Template version. Working version run by Supervisor).
* **`/data/01_clean_pool`**: Set of safe, obfuscated logs for your work.

### Source Characteristics (Sites)

We use data from three different environments, each with its own specifics:

1.  **Site Alpha (`site-alpha`):**
    * **Type:** High-Performance Cluster.
    * **Features:** Very powerful nodes (128 CPU / 750GB RAM).
    * **What to look for:** Ideal testing ground for FinOps. Many examples of **Infrastructure Mismatch** (resource usage imbalance) and **Missing Limits** in system services.

2.  **Site Beta (`site-beta`):**
    * **Type:** Unstable Production.
    * **Features:** Cluster under load with failure history.
    * **What to look for:** Main source of real errors. Look for **OOMKilled** (`consul` service) and **CrashLoopBackOff** here.

3.  **Site Gamma (`site-gamma`):**
    * **Type:** Standard Workload.
    * **Features:** Typical microservice architecture.
    * **What to look for:** Use as base for **synthesis** (Error Injection), since configurations here are relatively clean.

---

##  ML Stack: Training Recommendations

For thesis work, following stack is recommended:

### 1. Base Models
* **Llama 3 8B (Instruct):** *Recommended choice #1.* Best JSON generation quality.
* **Mistral 7B (v0.3):** Good alternative with large context window.

### 2. Training Library
Strongly recommended to use **Unsloth** (https://github.com/unslothai/unsloth).
* **Why:** Accelerates training 2-5x, allows model training on free GPUs (T4).

### 3. Execution Environment
* **Google Colab (Free/Pro):** Optimal for start.
* **Kaggle Kernels:** Alternative with free GPU T4 x2.

---

##  Workflow Algorithm

1.  **Data Preparation (Week 1):**
    * Obtain data from `/01_clean_pool`.
    * Follow instructions from File 04 (Gap Analysis & Synthesis).
    * Form `train.jsonl` (minimum 50-100 quality examples).

2.  **Fine-Tuning (Week 2):**
    * Use Unsloth for model fine-tuning on created dataset.

3.  **Validation (Week 3):**
    * Use methodology from File 05.
    * Compare Base Model and your Fine-Tuned Model responses (Math Consistency and JSON Valid Rate metrics).
