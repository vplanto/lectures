# Project Methodology: Kubernetes LLM Assistant

> 🇺🇦 [Українська версія](../methodology.md) | 🇬🇧 English version

**Trajectory:** Dataset Engineering Kit — from K8s problem taxonomy to fine-tuning an SRE assistant (Unsloth, JSON gold standard).

---

## Stages

| # | Document | Stage |
|---|----------|-------|
| 01 | [01_k8s_problem_taxonomy.md](../01_k8s_problem_taxonomy.md) | Scope: Break-fix vs Optimization |
| 02 | [02_detection_and_symptoms.md](../02_detection_and_symptoms.md) | Symptoms, kubectl, Prometheus |
| 03 | [03_dataset_output_templates.md](../03_dataset_output_templates.md) | Gold Standard JSON |
| 04 | [04_data_preparation_workflow.md](../04_data_preparation_workflow.md) | Obfuscation, Error Injection |
| 05 | [05_model_lifecycle_and_metrics.md](../05_model_lifecycle_and_metrics.md) | Validation, metrics |

## Integrity

- [DISCLAIMER](../../DISCLAIMER.md) · [sources.md](../sources.md)
- `01_clean_pool/` datasets are obfuscated; do not publish raw logs.
