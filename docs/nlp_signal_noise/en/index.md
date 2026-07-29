# Course: NLP in Technical Domains: From Bayes' Theorem to AI-SRE
## Topic: Signal in the Noise

> 🇺🇦 [Українська версія](../index.md) | 🇬🇧 English version

**Author:** Vitaliy Platonov

**Audience:** 2nd-3rd year students (Applied Mathematics, CS)

---

### Abstract
Why do engineers ignore 99% of monitoring system alerts?

This course examines the fundamental problem of "finding a needle in a haystack" through the lens of probability theory and modern NLP.

We'll journey from classical Bayes' formula (18th century) to BERT transformers (21st century) to create an "intelligent filter" for technical logs.

### Teaching Kit (NMK)

| | |
|---|---|
| **Methodology** | [Course trajectory](./methodology.md) |
| **Sources** | [Primary-source register](../sources.md) |
| **Declaration** | [Academic integrity](../../en/DISCLAIMER.md) |


---

### Course Structure (Navigation)

#### Block 1: Intuition and the Mathematical Trap
Introduction through a paradox that explains why intuition fails us when evaluating rare events (errors, diseases, attacks).

* **[00_the_bayesian_trap.md](./00_the_bayesian_trap.md)**
    * **Topic:** The Bayesian Trap (Recap of Veritasium video).
    * **Content:** Analysis of medical test example. Concepts of `False Positive` vs `False Negative`. Why a 99% accurate test is wrong 90% of the time if the disease is rare. The law of total probability.
    * **Materials:** Veritasium video, basic Python script for paradox simulation.

* **[01_noise_in_production.md](./01_noise_in_production.md)**
    * **Topic:** Industry problems: Logs, Spam, Fraud.
    * **Content:** Mapping medical problem to IT. "Disease" = "Critical database failure" (happens rarely). "Symptom" = "Error line in logs" (happens often). Why `grep "Error"` doesn't work. Concept of "Base Rate Fallacy" in DevOps (Alert Fatigue).

#### Block 2: Statistical Approach (Frequency-based)
Classical methods that work with word frequency (Bag of Words).

* **[02_math_setup_classification.md](./02_math_setup_classification.md)**
    * **Topic:** Mathematical formalization of classification task.
    * **Content:** Text as a set of tokens $X = \\{w_1, w_2, \\dots, w_n\\}$. Problem statement $P(\\text{Class} \\mid \\text{Features})$. Bayes' theorem: $P(\\text{Spam} \\mid \\text{Word}) = \\frac{P(\\text{Word} \\mid \\text{Spam})P(\\text{Spam})}{P(\\text{Word})}$.
    
* **[03_naive_bayes_deep_dive.md](./03_naive_bayes_deep_dive.md)**
    * **Topic:** Naive Bayes and its limitations.
    * **Content:** Why it's "naive" (independence assumption for words). Why it works for spam (words "viagra" and "casino" correlate), but poorly for logs ("Connection" and "Refused" are only important together).

#### Block 3: Geometric Approach (Semantic-based)
Transition from "word counting" to "meaning understanding". Using vector algebra.

* **[04_geometry_of_meaning.md](./04_geometry_of_meaning.md)**
    * **Topic:** From words to vectors (Embeddings).
    * **Content:** Synonym problem ("DB down" vs "Connection lost"). Vector spaces ($R^N$). Cosine distance as similarity measure. Word2Vec intuition: $King - Man + Woman = Queen$.

* **[05_bert_and_transformers.md](./05_bert_and_transformers.md)**
    * **Topic:** Transformers in technical analysis.
    * **Content:** BERT architecture (Bidirectional Encoder Representations). Self-Attention mechanism: how model understands context. Why BERT better detects anomalies in event sequences than Bayes. Using `[CLS]` token for incident classification.

#### Block 4: Практикум (Workshop)
Creating a working prototype.

* **[06_synthetic_chaos_generator.md](./06_synthetic_chaos_generator.md)**
    * **Topic:** Synthetic dataset generation.
    * **Content:** Real log confidentiality problem. How to generate "correct noise" and "rare anomalies" using Python (`Faker` library or LLM prompting). Creating imbalanced dataset (99% OK, 1% Error) to verify Bayesian Trap.

* **[07_implementation_workshop.md](./07_implementation_workshop.md)**
    * **Topic:** Building filtering pipeline.
    * **Content:** Live-coding session.
        1. **Baseline:** Naive Bayes implementation (scikit-learn) — achieving high accuracy but low Recall (missing anomalies).
        2. **Advanced:** BERT-based classifier implementation (HuggingFace) — improving contextual error detection.
        3. **Metric War:** Comparing Accuracy vs F1-score on imbalanced data.

* **[08_course_summary.md](./08_course_summary.md)**
    * **Topic:** Summary and future (LLM Agents).
    * **Content:** What we learned. Where to go next: RAG (Retrieval-Augmented Generation) — so AI not only classifies errors but also writes instructions to fix them (Runbook Automation).

#### Seminars: Filling Gaps Between Theory and Practice
Additional practical modules for deeper concept understanding.

* **[09_seminar_high_dimensional_geometry.md](./09_seminar_high_dimensional_geometry.md)**
    * **Topic:** High-dimensional geometry and visualization.
    * **Content:** Why it matters: BERT vectors have 768 dimensions. Students struggle to imagine such "value geometry". Dimensionality reduction methods (t-SNE, UMAP). Practice: Visualizing how log vectors "cluster" by incident types (e.g., separate cluster for Database errors, separate for Network timeouts).

* **[10_seminar_metrics_extreme_imbalance.md](./10_seminar_metrics_extreme_imbalance.md)**
    * **Topic:** Metrics under extreme class imbalance.
    * **Content:** Why it matters: Sources indicate Accuracy is misleading on imbalanced data. Deep dive into Precision-Recall Curves (PRC) compared to ROC-AUC. Why PRC is more informative for rare events (Base Rate < 0.1%). Mathematics: Deriving relationship between Base Rate and Precision through Bayes' theorem.

* **[11_seminar_efficient_nlp_quantization_distillation.md](./11_seminar_efficient_nlp_quantization_distillation.md)**
    * **Topic:** Efficient NLP: Quantization and Distillation.
    * **Content:** Why it matters: BERT is a large and expensive model (110M+ parameters), creating problems for real-time monitoring. Introduction to DistilBERT and methods for model size reduction without significant quality loss. Practice: Comparing log processing speed (inference time) between baseline BERT and optimized model.

* **[12_seminar_regex_vs_tokenizers.md](./12_seminar_regex_vs_tokenizers.md)**
    * **Topic:** Regular Expressions vs Tokenizers.
    * **Content:** Why it matters: Before text becomes a vector, it must be cleaned. Technical logs are "dirty" (IP addresses, hashes). Seminar on efficient log parsing and creating custom tokenizers that don't split important technical entities (e.g., java.lang.NullPointerException should be treated as one token).

#### Study Materials and Self-Learning Plan
Practical materials for the study group focusing on Case Study, synthetic data, trajectories for final presentation, and MCC as the "gold standard".

* **[study_materials.md](./study_materials.md)**
    * **Topic:** Study materials and self-learning plan.
    * **Content:**
        1. **Study Materials (Theory Block)** - Case Study: Alert Fatigue, synthetic chaos generator, MCC as "gold standard".
        2. **Self-Learning Plan** - Modules with practical tasks:
           - Module #1: Alert Fatigue and Base Rate Fallacy
           - Module #2: Synthetic Chaos Generator
           - Module #3: MCC vs F1-Score on imbalanced data
        3. **Cases for Final Presentation** - Three trajectories:
           - Option 1: Classical (Naive Bayes) - Log Spam Filter
           - Option 2: Semantic (BERT) - Anomaly Detection in Kubernetes Logs
           - Option 3: Optimization (DistilBERT) - Fast AI Monitoring
        4. **Practical MCC Application** - Comparing MCC vs F1-Score with practical code examples.
        5. **Topics for Course/Thesis Projects** - Research directions based on course materials.
