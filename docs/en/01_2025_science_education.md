# Research and Methodological Activities: Student Practical Training Focus

> 🇺🇦 [Українська версія](../01_2025_science_education.md) | 🇬🇧 English version

**Dear colleagues!**

Today I would like to share my experience and plans regarding research and methodological work, with a special emphasis on strengthening the practical component of our students' training. In the modern world, employers increasingly value not only theoretical knowledge but also the graduate's readiness to solve specific applied tasks.

### 1. Updating Educational Materials: From Theory to Practice

Analysis of student feedback revealed the need to make our educational materials more accessible and closely tied to practical tasks. Often theoretical foundations that seem clear to us are difficult to understand without a clear connection to real scenarios.

To address this, I engage **AI assistants as a tool for reviewing and improving teaching methodology**. Instead of traditional lectures, we jointly create **interactive workshops** where theory is immediately applied to solve tasks relevant to modern students. Using **Markdown** and **GitHub** significantly simplifies this process, allowing focus on content, easy material updates, and presenting them as a unified online resource.

This work is directly embodied in the **plan for preparing educational and methodological publications for 2026**. I am currently actively developing new **lecture notes and methodological guidelines** for key disciplines, based precisely on these updated principles.

### 2. Plan for Publishing Educational and Methodological Support (for 2026 academic year)

For the 2026 academic year, the following materials embodying an updated practice-oriented approach are planned for preparation and publication:

* **"Programming" Discipline:**
    * Methodological guidelines for laboratory work (covering C++ from basics to complex algorithms and OOP).
    * Updated lecture notes.
* **"Internet Technologies Fundamentals" Discipline:**
    * Methodological guidelines for a cycle of laboratory work on web development (HTML, CSS, JavaScript).
    * Lecture notes (including OSI model, transport (TCP/UDP) and application (HTTP/HTTPS/WS and others) layer protocols).
* **"Software Product Creation and Testing Technologies" Discipline:**
    * Lecture notes (development methodologies, software lifecycle, QA fundamentals, **with focus on software design**).
    * Methodological guidelines for course project design (with Java examples and Android app development).
* **"Construction and Analysis of Decentralized Systems" Discipline:**
    * Lecture notes (architectures, consensus, blockchain).
    * Methodological guidelines for laboratory work on DApp development.

### 3. Student Research Work: Solving Applied Problems

Practical focus is also key in managing student research work. I strive to direct them to topics where deep theoretical knowledge is applied to solve clearly defined, relevant problems.

**Examples of current student projects:**

1.  **Agent-based modeling of transportation flows for evacuation plan optimization:**
    * **Essence:** Development of a model for analyzing mass population evacuation processes. **The goal is to create a model capable of analyzing any city in Ukraine based on open data (OpenStreetMap), simulating evacuation, identifying bottlenecks, and evaluating the effectiveness of managed (controlled) evacuation scenarios.**
    * **Focus:** The system performs **predictive analysis** (congestion forecasting) and **prescriptive analysis** (route change recommendations generation).
    * **Clarity:** Results are presented visually (congestion map) and as clear recommendations.

2.  **Applying fine-tuned language models for Kubernetes cluster state analysis:**
    * **Essence:** Creating an "intelligent assistant" for DevOps/SRE engineers. Uses a large language model **additionally trained (fine-tuned)** on specific Kubernetes data (logs, metrics).
    * **Focus:** The model **interprets** technical data, identifies problem causes, and provides specific recommendations in structured format.
    * **Clarity:** Transforming complex, unstructured data into clear conclusions and advice.

3.  **Hybrid method for semantic filtering of non-informative records in system logs:**
    * **Essence:** Developing a system for automatic identification of "information noise" from logs. Uses a combined approach: modern language understanding models (like **BERT**) for **semantic vectorization**, **clustering algorithms** for grouping similar messages, cluster **entropy** analysis, and **Naive Bayes** classifier.
    * **Focus:** Effective detection and removal of **massive, repetitive, insignificant** messages, allowing engineers to focus on important events.
    * **Clarity:** System operation is explained through visualization of log groups (using **dimensionality reduction** methods) and analysis of specific examples of filtered records.

4.  **Comparative analysis of algorithms for anomaly detection in IT system metric time series:**
    * **Essence:** Comprehensive comparison of six methods (statistical, ML, clustering, RNN) for anomaly detection in system metrics (CPU, RAM, etc.).
    * **Focus:** Testing algorithms on three dataset types (controlled, synthetic, real) to evaluate their sensitivity and reliability under different conditions. Work showed **RNN** advantages for analyzing complex temporal dependencies.
    * **Further development:** This work became the foundation for master's research focused on **interpreting** the RNN "black box" using SHAP/LIME methods and building a **hybrid** monitoring system.

Additionally, based on **last year's** student work results, articles are currently being prepared for publication, emphasizing the continuity of the research process and the importance of documenting obtained results.

### **Conclusions**

Updating educational materials with a focus on clarity and practice, as well as managing student research work aimed at solving specific applied tasks, are important components of training competent and in-demand specialists in IT and the modern world.

**Thank you for your attention!**

---

## Update: December 2025. Strategic Adjustments (Stanford CS230 Insights)
**Basis:** Market trend analysis Q4 2025 (Andrew Ng, Laurence Moroney).

Based on recent data on the stagnation of the "Big Model Growth" approach and the transition to "Agentic Workflows," I am making changes to the methodology for managing student projects.

### 1. Project #2 Pivot: From Fine-Tuning to Agentic RAG
Current project *"Applying fine-tuned language models for K8s"* requires architectural change.
* **Problem:** Fine-tuning is expensive and static. The model "forgets" new Kubernetes API versions.
* **New vector:** **Agentic Workflows**.
    * Instead of training a model, students develop an **Agent** that has access to tools (Tools: `kubectl`, `Prometheus API`).
    * **Skill:** The student learns to write not prompts, but a **Control Loop** (Intent $\to$ Plan $\to$ Tool Use $\to$ Reflection), which is much more relevant to the 2025 job market.

### 2. New Pedagogical Tool: "Hardware Constraint"
To prepare students for the reality of **"Small AI"** (Edge/On-device), I'm introducing artificial constraints for course projects:
* **Cloud ban:** The solution must work locally on laptop CPU (or integrated NPU).
* **Goal:** Force students to understand quantization (GGUF Quantization), memory optimization, and latency, instead of blindly using unlimited OpenAI/Cloud resources.

### 3. Countering "Vibe Coding" in the Educational Process
A mandatory work defense stage is introduced: **"Verification Audit"**.
The student must not just show working code (which Copilot could generate), but prove its correctness through:
* Formal complexity analysis ($O$-notation).
* Stress tests of boundary values.
* Explanation of each line of generated code.
* **Principle:** "If you can't explain it — you didn't pass, even if it works".
