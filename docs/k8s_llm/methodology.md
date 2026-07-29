# Методологія проєкту: Kubernetes LLM Assistant

> 🇺🇦 Українська версія | 🇬🇧 [English version](./en/methodology.md)

**Траєкторія:** Dataset Engineering Kit — від таксономії K8s-проблем до fine-tuning SRE-асистента (Unsloth, JSON gold standard).

---

## Етапи

| № | Документ | Етап |
|---|----------|------|
| 01 | [01_k8s_problem_taxonomy.md](./01_k8s_problem_taxonomy.md) | Scope: Break-fix vs Optimization |
| 02 | [02_detection_and_symptoms.md](./02_detection_and_symptoms.md) | Симптоми, kubectl, Prometheus |
| 03 | [03_dataset_output_templates.md](./03_dataset_output_templates.md) | Gold Standard JSON |
| 04 | [04_data_preparation_workflow.md](./04_data_preparation_workflow.md) | Обфускація, Error Injection |
| 05 | [05_model_lifecycle_and_metrics.md](./05_model_lifecycle_and_metrics.md) | Валідація, метрики |

## Доброчесність

- [DISCLAIMER](../DISCLAIMER.md) · [sources.md](./sources.md)
- Датасети `01_clean_pool/` — обфусковані; не публікувати raw-логи.
