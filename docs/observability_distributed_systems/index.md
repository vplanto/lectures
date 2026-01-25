# Курс: Спостережуваність (Observability) та інтелектуальна діагностика розподілених систем
## Тема: Від моніторингу "мертвих" метрик до розуміння Unknown Unknowns

**Автор:** Віталій Платонов
**Версія:** 1.0
**Статус:** Draft
**Аудиторія:** Студенти 3-4 курсу, SRE-інженери, DevOps Architects

---

### Анотація курсу
Традиційний моніторинг у Kubernetes не справляється з High Cardinality. Цей курс про **Спостережуваність (Observability)** — коли система може відповісти на питання, які ви не передбачили заздалегідь. Ми поєднуємо класичний OS-стек (Prometheus, Loki, Graylog) з методами AI-діагностики та математичною теорією росту (Growth Theory) для побудови автономних SRE-агентів.

---

### Структура курсу (Навігація)

#### Блок 0: Маніфест Спостережуваності (The Problem)
Вступ у проблематику через деконструкцію класичних підходів.

* **[00_observability_manifesto.md](./00_observability_manifesto.md)**
    * **Тема:** Моніторинг vs Спостережуваність (Observability).
    * **Зміст:** Проблема "Unknown Unknowns". Чому дашборди брешуть. Перехід від перевірки працездатності до дослідження станів системи.
* **[01_hardware_vs_software_context.md](./01_hardware_vs_software_context.md)**
    * **Тема:** Прокляття кардинальності: IT-системи vs Hardware Fabrics.
    * **Зміст:** Математика вибуху простору станів $O(2^N)$. Чому 100,000 метрик від заліза менш корисні за один структурований event додатка.

#### Блок 1: Математика стійкості та сигналу
Використання існуючого математичного апарату для фільтрації шуму та моделювання розпаду.

* **[02_noise_reduction_prerequisites.md](./02_noise_reduction_prerequisites.md)**
    * **Тема:** Боротьба з Втома від сповіщень (Alert Fatigue) (Bayesian Logic).
    * **Зміст:** Референс на [00_the_bayesian_trap.md](../nlp_signal_noise/00_the_bayesian_trap.md). Розрахунок ймовірності реального інциденту в шумному середовищі.
* **[03_p99_and_tail_latency.md](./03_p99_and_tail_latency.md)**
    * **Тема:** Математика квантилів та довгих хвостів.
    * **Зміст:** Чому Average — це ілюзія. Використання MCC як золотого стандарту для незбалансованих даних.
* **[04_growth_theory_resilience.md](./04_growth_theory_resilience.md)**
    * **Тема:** Теорія росту в мікросервісах (Growth Theory).
    * **Зміст:** Математика стійкості: S-криві vs Лінійний розпад. Моделювання порушень SLA як стохастичного процесу.

#### Блок 2: Технологічний стек (Implementation)
Побудова Спостережуваність (Observability) Pipeline на базі Open Source інструментів.

* **[05_prometheus_tsdb_internals.md](./05_prometheus_tsdb_internals.md)**
    * **Тема:** TSDB та PromQL під високим навантаженням.
    * **Зміст:** Векторні операції та ефективне зберігання метрик. Проблема High Cardinality в Prometheus.
* **[06_event_driven_logging_loki.md](./06_event_driven_logging_loki.md)**
    * **Тема:** Логи як події (Loki/Graylog).
    * **Зміст:** Перехід від текстового пошуку до семантичного індексування. Кореляція TraceID та Log Lines.

#### Блок 3: AI-SRE та Аналіз (Diagnostics)
Застосування ML та NLP для автоматичної інтерпретації інцидентів.

* **[07_semantic_log_analysis.md](./07_semantic_log_analysis.md)**
    * **Тема:** Семантичний пошук інцидентів (Embeddings).
    * **Зміст:** Референс на [04_geometry_of_meaning.md](../nlp_signal_noise/04_geometry_of_meaning.md). Перетворення логів у вектори для пошуку схожих проблем у минулому.
* **[08_bert_for_log_anomalies.md](./08_bert_for_log_anomalies.md)**
    * **Тема:** Трансформери в SRE (Contextual Understanding).
    * **Зміст:** Референс на [05_bert_and_transformers.md](../nlp_signal_noise/05_bert_and_transformers.md). Чому BERT бачить зв'язок між збоями, який пропускає статистика.
* **[09_xai_for_incident_response.md.md](./09_xai_for_incident_response.md.md)**
    * **Тема:** Пояснювальний ШІ та Першопричина (Root Cause) Analysis.
    * **Зміст:** Застосування Shapley Values та контрфактуального аналізу для генерації рецептів виправлення (Prescription).

#### Практикум (Workshop)
Створення автономного діагноста.

* **[10_chaos_engineering_lab.md](./10_chaos_engineering_lab.md)**
    * **Тема:** Python-генератор синтетичного хаосу.
    * **Зміст:** Імітація каскадних збоїв та OOM-killing для тестування систем моніторингу.
* **[11_llm_incident_diagnostics.md](./11_llm_incident_diagnostics.md)**
    * **Тема:** RAG-архітектура для Smart Runbooks.
    * **Зміст:** Поєднання бази знань інцидентів з LLM для автоматичної генерації рекомендацій інженерам.

---
**[material_study.md](./material_study.md)** — Гайд для самостійного вивчення.